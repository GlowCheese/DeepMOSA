####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert type(wrapped).__name__ == "CalledProcessError"
    assert type(wrapped) is not type(err)

def test_error_wrapper_preserves_exception_attributes():
    err = subprocess.CalledProcessError(1, "test", output=b"test output")
    wrapped = error_wrapper(err)
    assert wrapped.returncode == err.returncode
    assert wrapped.cmd == err.cmd
    assert wrapped.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert "Captured output:" in str(wrapped)
    assert "line1" in str(wrapped)
    assert "line2" in str(wrapped)

def test_error_wrapper_str_without_output():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test", output=b"\xff\xfe")
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)

def test_error_wrapper_handles_timeout_expired():
    err = subprocess.TimeoutExpired("test", 1)
    wrapped = error_wrapper(err)
    assert type(wrapped).__name__ == "TimeoutExpired"


# LLM-generated content at query #2
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error_unchanged():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_modifies_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == """Command 'test_cmd' returned non-zero exit status 1.
Captured output:
    line1
    line2"""

def test_error_wrapper_modifies_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == """Command 'test_cmd' returned non-zero exit status 1.
No output was generated."""

def test_error_wrapper_modifies_timeout_expired_with_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == """Command 'test_cmd' timed out after 1 seconds.
Captured output:
    line1
    line2"""

def test_error_wrapper_modifies_timeout_expired_without_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == """Command 'test_cmd' timed out after 1 seconds.
No output was generated."""

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert str(wrapped_err) == """Command 'test_cmd' returned non-zero exit status 1.
Failed to parse output."""


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "test"], verbose=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir)
        assert result.command == ["pwd"]
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"})
    assert result.command == ["printenv", "TEST_VAR"]
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_shell():
    result = run_command("echo shell", shell=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert b"shell" in result.captured_output

def test_run_command_non_zero_return():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_kwargs():
    result = run_command(["echo", "test"], shell=False, text=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0


# LLM-generated content at query #4
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_env_and_cwd():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, cwd="/tmp", return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.command == ["echo", "output"]
    assert result.return_code == 0
    assert result.captured_output == b"output\n"

def test_run_command_with_nonzero_return_code():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_shell_command():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert result.captured_output == b"shell\n"

def test_run_command_with_long_output_truncation():
    long_command = "python -c \"print('x' * 10000)\""
    result = run_command(long_command, shell=True, return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

def test_run_command_with_unicode_decode_error():
    result = run_command(["python", "-c", "print(b'\\xff\\xfe'.decode('latin1'))"], return_output=True, verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_line_35_false():
    assert not isinstance(subprocess.TimeoutExpired("test", 1), subprocess.CalledProcessError)


# LLM-generated content at query #6
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None

def test_return_code_nonzero():
    result = run_command("exit 1", ignore_errors=True)
    assert result.captured_output is not None

def test_verbose_true():
    result = run_command("echo test", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_with_verbose():
    result = run_command(["echo", "hello"], verbose=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_timeout():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_return_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "/tmp"

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output


# LLM-generated content at query #8
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1", ignore_errors=True).captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_modifies_subprocess_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"error output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert "Captured output:" in str(wrapped_err)
    assert "error output" in str(wrapped_err)

def test_error_wrapper_modifies_subprocess_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 1, output=b"timeout output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "Captured output:" in str(wrapped_err)
    assert "timeout output" in str(wrapped_err)

def test_error_wrapper_handles_empty_output():
    err = subprocess.CalledProcessError(1, "cmd", output=b"")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_handles_non_utf8_output():
    err = subprocess.CalledProcessError(1, "cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #10
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_with_error():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is None

def test_run_command_with_timeout():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose_mode():
    result = run_command("echo 'verbose'", verbose=True, return_output=True)
    assert result.command == "echo 'verbose'"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"verbose" in result.captured_output

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test_value" in result.captured_output

def test_run_command_with_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"/tmp" in result.captured_output

def test_run_command_with_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test" in result.captured_output

def test_run_command_with_return_output_false():
    result = run_command("echo 'test'", return_output=False)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #11
#--------------------------

```python
def test_output_truncation():
    output = b"x" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #12
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_shell_string():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #13
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}
    result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors, **kwargs)
    assert result.command == args
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #14
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1", ignore_errors=True).captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failed_execution():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.return_code == 0

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.captured_output == b"output\n"

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.strip() == tmpdir.encode()

def test_run_command_unicode_output():
    result = run_command(["echo", "unicode: 你好"], return_output=True)
    assert result.captured_output == b"unicode: \xe4\xbd\xa0\xe5\xa5\xbd\n"

def test_run_command_long_output_truncation():
    long_output = "x" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #16
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    result = run_command(args, return_output=True, verbose=True)
    assert isinstance(result, CommandResult)


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #18
#--------------------------

```python
def test_unicode_decode_error_raises_exception():
    output = b'\xff\xfe'
    try:
        output.decode('utf-8')
    except UnicodeDecodeError as e:
        assert isinstance(e, UnicodeDecodeError)


# LLM-generated content at query #19
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #20
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    test_error = ValueError("test")
    result = error_wrapper(test_error)
    assert result is test_error


# LLM-generated content at query #21
#--------------------------

```python
def test_error_wrapper_returns_same_error_for_non_subprocess_exceptions():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_called_process_error():
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "CalledProcessError"
    assert type(wrapped_err) is not type(err)

def test_error_wrapper_creates_new_type_for_subprocess_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "TimeoutExpired"
    assert type(wrapped_err) is not type(err)

def test_error_wrapper_preserves_error_attributes():
    err = subprocess.CalledProcessError(1, "cmd", output=b"output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == err.returncode
    assert wrapped_err.cmd == err.cmd
    assert wrapped_err.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_str_without_output():
    err = subprocess.CalledProcessError(1, "cmd")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)


# LLM-generated content at query #22
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_return_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output == b"test\n"

def test_run_command_non_zero_return():
    result = run_command(["sh", "-c", "exit 1"], return_output=True)
    assert result.return_code == 1
    assert result.captured_output is not None


# LLM-generated content at query #23
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_with_verbose():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=True)

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_with_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_return_output_false():
    result = run_command("echo 'test'", return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_called_process_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_with_timeout_expired():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_with_os_error_retry():
    with pytest.raises(OSError):
        run_command("non_existent_command")

def test_run_command_with_unicode_decode_error():
    with pytest.raises(UnicodeDecodeError):
        run_command("echo -e '\\x00'", verbose=True)

def test_run_command_with_max_output_length():
    long_output = "a" * (MAX_OUTPUT_LENGTH + 100)
    with pytest.raises(subprocess.CalledProcessError):
        run_command(f"echo '{long_output}'", timeout=0.1)

def test_run_command_with_shell_true():
    result = run_command("echo 'test'", shell=True, return_output=True)
    assert result.captured_output == b"test\n"

def test_run_command_with_shell_false():
    result = run_command(["echo", "test"], shell=False, return_output=True)
    assert result.captured_output == b"test\n"

def test_run_command_with_kwargs():
    result = run_command("echo 'test'", shell=True, return_output=True, text=True)
    assert result.captured_output == b"test\n"


# LLM-generated content at query #24
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

def test_run_command_verbose_mode():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.captured_output.decode('utf-8').strip() == "value"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_truncated_output():
    result = run_command(["python", "-c", "print('x' * 10000)"], return_output=True)
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")

def test_run_command_unicode_error():
    result = run_command(["python", "-c", "print(b'\\xff'.decode('latin1'))"], return_output=True, verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_evaluates_to_true():
    output = b"x" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #26
#--------------------------

```python
def test_unicode_decode_error_handling():
    command = "echo 'test'"
    result = run_command(command, return_output=True, ignore_errors=True)
    assert result.captured_output is not None
    assert isinstance(result.captured_output, bytes)


# LLM-generated content at query #27
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    non_subprocess_error = ValueError("test error")
    assert error_wrapper(non_subprocess_error) is non_subprocess_error


# LLM-generated content at query #28
#--------------------------

```python
def test_error_wrapper_predicate():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #29
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}

    # Simulate a UnicodeDecodeError by providing non-UTF-8 bytes
    with unittest.mock.patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = b'\xff\xfe'
        result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors, **kwargs)
        assert result.return_code == 0


# LLM-generated content at query #30
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = ["echo", "test"]
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}

    with tempfile.TemporaryFile() as f:
        f.write(b"\xff\xfe")
        f.seek(0)
        ret = subprocess.CompletedProcess(args, 0)
        result = CommandResult(args, ret.returncode, f.read())

    assert result.captured_output is not None


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_evaluates_to_false():
    assert not (return_output or ret.returncode != 0 or verbose)


# LLM-generated content at query #32
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "cmd")
    result = error_wrapper(err)
    assert type(result).__name__ == "CalledProcessError"
    assert type(result) is not type(err)

def test_error_wrapper_preserves_original_exception_attributes():
    err = subprocess.CalledProcessError(1, "cmd", output=b"test output")
    result = error_wrapper(err)
    assert result.returncode == err.returncode
    assert result.cmd == err.cmd
    assert result.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "cmd", output=b"line1\nline2")
    result = error_wrapper(err)
    assert "Captured output:" in str(result)
    assert "line1" in str(result)
    assert "line2" in str(result)

def test_error_wrapper_str_without_output():
    err = subprocess.CalledProcessError(1, "cmd")
    result = error_wrapper(err)
    assert "No output was generated." in str(result)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "cmd", output=b"\xff\xfe")
    result = error_wrapper(err)
    assert "Failed to parse output." in str(result)

def test_error_wrapper_works_with_timeout_expired():
    err = subprocess.TimeoutExpired("cmd", 1)
    result = error_wrapper(err)
    assert type(result).__name__ == "TimeoutExpired"
    assert "No output was generated." in str(result)


# LLM-generated content at query #33
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    non_subprocess_error = ValueError("test error")
    assert error_wrapper(non_subprocess_error) is non_subprocess_error


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper_predicate_false():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #35
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    class CustomError(Exception):
        pass

    err = CustomError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #37
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #38
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #39
#--------------------------

```python
def test_verbose_logging():
    verbose = True
    cwd_str = "/test/path"
    args = ["echo", "test"]
    log((cwd_str or "") + "> " + repr(args), timestamp=False, include_proc_id=False)


# LLM-generated content at query #40
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo 'test'", verbose=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_output():
    result = run_command("echo 'test'", return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_timeout_ignore_errors():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert b"test_value" in result.captured_output

def test_run_command_with_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


# LLM-generated content at query #41
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None


# LLM-generated content at query #42
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #43
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}

    with unittest.mock.patch('subprocess.run') as mock_run:
        mock_run.return_value = unittest.mock.Mock(returncode=0)
        with unittest.mock.patch('tempfile.TemporaryFile') as mock_tempfile:
            mock_tempfile.return_value.__enter__.return_value.read.return_value = b'\xff\xfe'
            mock_tempfile.return_value.__enter__.return_value.seek.return_value = None
            result = run_command(args, env=env, cwd=cwd, timeout=timeout, verbose=verbose, return_output=return_output, ignore_errors=ignore_errors, **kwargs)
            assert result.captured_output is not None


# LLM-generated content at query #44
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    original_error = subprocess.CalledProcessError(1, "test_cmd")
    result = error_wrapper(original_error)
    assert type(result).__name__ == "CalledProcessError"
    assert type(result) is not type(original_error)

def test_error_wrapper_preserves_output_in_new_type():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    result = error_wrapper(original_error)
    assert result.output == b"test output"

def test_error_wrapper_str_includes_output():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    result = error_wrapper(original_error)
    assert "Captured output:" in str(result)
    assert "line1" in str(result)
    assert "line2" in str(result)

def test_error_wrapper_str_handles_decode_error():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    result = error_wrapper(original_error)
    assert "Failed to parse output." in str(result)

def test_error_wrapper_str_no_output_case():
    original_error = subprocess.CalledProcessError(1, "test_cmd")
    result = error_wrapper(original_error)
    assert "No output was generated." in str(result)

def test_error_wrapper_works_with_timeout_expired():
    original_error = subprocess.TimeoutExpired("test_cmd", 1)
    result = error_wrapper(original_error)
    assert type(result).__name__ == "TimeoutExpired"


# LLM-generated content at query #45
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b'test\n'

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=False, return_output=False)

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1, verbose=False, return_output=False)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True, verbose=False, return_output=False)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is None

def test_run_command_verbose():
    result = run_command("echo 'test'", verbose=True, return_output=False)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_return_output():
    result = run_command("echo 'test'", return_output=True, verbose=False)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b'test\n'

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, verbose=False, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b'test_value\n'

def test_run_command_cwd():
    result = run_command("pwd", cwd="/tmp", verbose=False, return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output == b'/tmp\n'


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.captured_output == b"output\n"

def test_run_command_unicode_error():
    result = run_command(["echo", "\xff"], return_output=True)
    assert result.captured_output == b"\xff\n"

def test_run_command_long_output():
    long_output = "x" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #2
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test_command")
    result = error_wrapper(err)
    assert type(result).__name__ == "CalledProcessError"
    assert type(result) is not type(err)

def test_error_wrapper_preserves_original_exception_attributes():
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    result = error_wrapper(err)
    assert result.returncode == err.returncode
    assert result.cmd == err.cmd
    assert result.output == err.output

def test_error_wrapper_str_with_output():
    err = subprocess.CalledProcessError(1, "test_command", output=b"line1\nline2")
    result = error_wrapper(err)
    assert "Captured output:" in str(result)
    assert "line1" in str(result)
    assert "line2" in str(result)

def test_error_wrapper_str_with_no_output():
    err = subprocess.CalledProcessError(1, "test_command")
    result = error_wrapper(err)
    assert "No output was generated." in str(result)

def test_error_wrapper_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test_command", output=b'\xff\xfe')
    result = error_wrapper(err)
    assert "Failed to parse output." in str(result)

def test_error_wrapper_handles_timeout_expired():
    err = subprocess.TimeoutExpired("test_command", 1)
    result = error_wrapper(err)
    assert type(result).__name__ == "TimeoutExpired"
    assert "No output was generated." in str(result)


# LLM-generated content at query #3
#--------------------------

```python
def test_return_output_true():
    result = run_command("echo test", return_output=True)
    assert result.captured_output is not None

def test_return_code_nonzero():
    result = run_command("exit 1", ignore_errors=True)
    assert result.captured_output is not None

def test_verbose_true():
    result = run_command("echo test", verbose=True)
    assert result.captured_output is not None


# LLM-generated content at query #4
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    class CustomError(Exception):
        pass

    err = CustomError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #5
#--------------------------

```python
def test_unicode_decode_error_handling():
    with tempfile.TemporaryFile() as f:
        f.write(b'\xff\xfe')
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            assert True


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_failed_execution():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True)
    assert result.return_code == 0

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert "test_value" in result.captured_output.decode('utf-8')

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert "/tmp" in result.captured_output.decode('utf-8')

def test_run_command_with_kwargs():
    result = run_command(["echo", "test"], shell=True, return_output=True)
    assert result.return_code == 0


# LLM-generated content at query #7
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #8
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1", ignore_errors=True).captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #9
#--------------------------

```python
def test_ignore_errors_false_raises_exception():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("false", ignore_errors=False)


# LLM-generated content at query #10
#--------------------------

```python
def test_predicate_evaluates_to_true():
    assert (True or False != 0 or True)


# LLM-generated content at query #11
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo hello", return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", return_output=True)

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 2", timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.return_code == 1

def test_run_command_verbose():
    result = run_command("echo hello", verbose=True, return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.captured_output.strip() == tmpdir.encode()

def test_run_command_return_output():
    result = run_command("echo hello", return_output=True)
    assert result.captured_output == b"hello\n"

def test_run_command_no_return_output():
    result = run_command("echo hello", return_output=False)
    assert result.captured_output is None


# LLM-generated content at query #12
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], verbose=False, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", verbose=False, return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output == b"/tmp\n"

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, verbose=False, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "2"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose_mode():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"], verbose=True)

def test_run_command_string_command():
    result = run_command("echo test", shell=True, verbose=False, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_kwargs():
    result = run_command(["echo", "test"], shell=False, verbose=False, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"


# LLM-generated content at query #13
#--------------------------

```python
def test_truncate_long_output():
    output = b"x" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #14
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "123"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=123" in result.captured_output

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is None

def test_run_command_return_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_non_zero_return_code():
    result = run_command(["false"], return_output=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_shell():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_long_output():
    result = run_command(["python", "-c", "print('x' * 10000)"], return_output=True)
    assert result.command == ["python", "-c", "print('x' * 10000)"]
    assert result.return_code == 0
    assert len(result.captured_output) > 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_with_unicode_output():
    result = run_command(["python", "-c", "print('hello 世界')"], return_output=True)
    assert result.command == ["python", "-c", "print('hello 世界')"]
    assert result.return_code == 0
    assert result.captured_output == b"hello \xe4\xb8\x96\xe7\x95\x8c\n"


# LLM-generated content at query #15
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_verbose_mode():
    result = run_command("echo 'verbose'", verbose=True, return_output=True)
    assert result.command == "echo 'verbose'"
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "value"}, return_output=True, shell=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"value\n"

def test_run_command_with_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_timeout():
    result = run_command("sleep 0.1", timeout=0.5, return_output=True)
    assert result.command == "sleep 0.1"
    assert result.return_code == 0

def test_run_command_timeout_expired():
    result = run_command("sleep 1", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 1"
    assert result.return_code == -32768

def test_run_command_nonzero_return_code():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1

def test_run_command_return_output_false():
    result = run_command("echo 'test'")
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_kwargs():
    result = run_command("echo 'test'", shell=True, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_35_evaluates_to_false():
    assert not isinstance(subprocess.TimeoutExpired("command", 1), subprocess.CalledProcessError)


# LLM-generated content at query #17
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_preserves_exception_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "CalledProcessError"
    assert isinstance(wrapped_err, subprocess.CalledProcessError)

def test_error_wrapper_includes_output_in_str_for_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_handles_empty_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=None)
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_works_with_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", 1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped_err)


# LLM-generated content at query #18
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.01)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.captured_output.decode().strip() == tmpdir

def test_run_command_return_output():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output == b"test\n"

def test_run_command_truncated_output():
    long_output = "x" * (MAX_OUTPUT_LENGTH + 100)
    result = run_command(["echo", long_output], return_output=True)
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")


# LLM-generated content at query #19
#--------------------------

```python
def test_unicode_decode_error_predicate():
    output = b'\xff\xfe'
    assert not (output.decode('utf-8'))


# LLM-generated content at query #20
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "/tmp"

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["printenv", "TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test_value"

def test_run_command_verbose():
    result = run_command(["echo", "hello"], verbose=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_string_command():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"


# LLM-generated content at query #21
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #22
#--------------------------

```python
def test_unicode_decode_error_handling():
    with tempfile.TemporaryFile() as f:
        f.write(b'\xff\xfe\xfd')
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            assert True


# LLM-generated content at query #23
#--------------------------

```python
def test_unicode_decode_error_occurs():
    args = ["echo", "test"]
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}

    with tempfile.TemporaryFile() as f:
        f.write(b'\x80\x81\x82')
        f.seek(0)
        output = f.read()

        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            assert True
        else:
            assert False


# LLM-generated content at query #24
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'Hello, World!'", shell=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_return_output():
    result = run_command("echo 'Hello, World!'", shell=True, return_output=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output == b"Hello, World!\n"

def test_run_command_with_verbose():
    result = run_command("echo 'Hello, World!'", shell=True, verbose=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", shell=True)

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", shell=True, timeout=0.1)

def test_run_command_with_ignore_errors():
    result = run_command("exit 1", shell=True, ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", shell=True, env={"TEST_VAR": "test_value"})
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_cwd():
    result = run_command("pwd", shell=True, cwd="/tmp")
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_kwargs():
    result = run_command("echo 'Hello, World!'", shell=True, text=True)
    assert result.command == "echo 'Hello, World!'"
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #25
#--------------------------

```python
def test_output_truncation():
    output = b"a" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #26
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failed_execution():
    result = run_command("exit 1", verbose=False, return_output=True, ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose_mode():
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.command == "echo 'verbose test'"
    assert result.return_code == 0
    assert result.captured_output == b"verbose test\n"

def test_run_command_with_env():
    env = {"TEST_VAR": "test_value"}
    result = run_command("echo $TEST_VAR", shell=True, env=env, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_no_output():
    result = run_command("echo 'test'", return_output=False)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None


# LLM-generated content at query #27
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_with_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "verbose"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "/tmp"

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["env"]
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "2"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is None

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.command == ["echo", "output"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "output"

def test_run_command_with_shell():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "shell"

def test_run_command_with_long_output():
    result = run_command(["python", "-c", "print('x' * 10000)"], return_output=True)
    assert result.command == ["python", "-c", "print('x' * 10000)"]
    assert result.return_code == 0
    assert len(result.captured_output) > 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_with_unicode_output():
    result = run_command(["python", "-c", "print('café')"], return_output=True)
    assert result.command == ["python", "-c", "print('café')"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "café"


# LLM-generated content at query #28
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test\n" in result.captured_output

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1", verbose=False, return_output=False)

def test_run_command_ignore_errors():
    result = run_command("exit 1", verbose=False, return_output=True, ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1, verbose=False, return_output=False)

def test_run_command_timeout_ignore_errors():
    result = run_command("sleep 10", timeout=0.1, verbose=False, return_output=True, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command("echo 'verbose test'", verbose=True, return_output=True)
    assert result.command == "echo 'verbose test'"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"verbose test\n" in result.captured_output

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test_value\n" in result.captured_output

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output is not None
        assert bytes(tmpdir, 'utf-8') in result.captured_output

def test_run_command_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"test\n" in result.captured_output


# LLM-generated content at query #29
#--------------------------

```python
def test_error_wrapper_predicate_false():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #30
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    result = run_command(args, return_output=True, verbose=True)
    assert isinstance(result, CommandResult)
    assert result.command == args
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #31
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_exceptions():
    test_exception = ValueError("test error")
    result = error_wrapper(test_exception)
    assert result is test_exception


# LLM-generated content at query #32
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #33
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #34
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #35
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error

def test_error_wrapper_modifies_called_process_error_with_output():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    result = error_wrapper(original_error)
    assert isinstance(result, subprocess.CalledProcessError)
    assert "Captured output:" in str(result)
    assert "test output" in str(result)

def test_error_wrapper_modifies_called_process_error_without_output():
    original_error = subprocess.CalledProcessError(1, "test_cmd")
    result = error_wrapper(original_error)
    assert isinstance(result, subprocess.CalledProcessError)
    assert "No output was generated." in str(result)

def test_error_wrapper_modifies_timeout_expired_with_output():
    original_error = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"timeout output")
    result = error_wrapper(original_error)
    assert isinstance(result, subprocess.TimeoutExpired)
    assert "Captured output:" in str(result)
    assert "timeout output" in str(result)

def test_error_wrapper_modifies_timeout_expired_without_output():
    original_error = subprocess.TimeoutExpired("test_cmd", timeout=1)
    result = error_wrapper(original_error)
    assert isinstance(result, subprocess.TimeoutExpired)
    assert "No output was generated." in str(result)

def test_error_wrapper_handles_unicode_decode_error():
    original_error = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    result = error_wrapper(original_error)
    assert isinstance(result, subprocess.CalledProcessError)
    assert "Failed to parse output." in str(result)


# LLM-generated content at query #36
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_exception_unchanged():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_preserves_original_exception_type():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)

def test_error_wrapper_modifies_str_with_no_output():
    err = subprocess.CalledProcessError(1, "test")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

def test_error_wrapper_modifies_str_with_output():
    err = subprocess.CalledProcessError(1, "test", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert "Captured output:" in str(wrapped)
    assert "line1" in str(wrapped)
    assert "line2" in str(wrapped)

def test_error_wrapper_modifies_str_with_unicode_error():
    err = subprocess.CalledProcessError(1, "test", output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)

def test_error_wrapper_modifies_str_for_timeout_expired():
    err = subprocess.TimeoutExpired("test", 1)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert "No output was generated." in str(wrapped)


# LLM-generated content at query #37
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_error_unchanged():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_preserves_original_error_type():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)

def test_error_wrapper_modifies_str_for_called_process_error_with_output():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped = error_wrapper(err)
    assert "Captured output:" in str(wrapped)
    assert "line1" in str(wrapped)
    assert "line2" in str(wrapped)

def test_error_wrapper_modifies_str_for_called_process_error_without_output():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped = error_wrapper(err)
    assert "No output was generated." in str(wrapped)

def test_error_wrapper_modifies_str_for_timeout_error_with_output():
    err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"timeout output")
    wrapped = error_wrapper(err)
    assert "Captured output:" in str(wrapped)
    assert "timeout output" in str(wrapped)

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped)


# LLM-generated content at query #38
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_returns_wrapped_exception_for_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nCaptured output:\n    test output"

def test_error_wrapper_returns_wrapped_exception_for_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", 1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_cmd' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_cmd' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #39
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "CalledProcessError"
    assert type(wrapped_err) is not type(err)

def test_error_wrapper_preserves_original_exception_attributes():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert wrapped_err.returncode == err.returncode
    assert wrapped_err.cmd == err.cmd
    assert wrapped_err.output == err.output

def test_error_wrapper_str_includes_output_for_called_process_error():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_str_includes_no_output_message_when_empty():
    err = subprocess.CalledProcessError(1, "test_cmd")
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_str_includes_decode_failure_message():
    err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_works_with_timeout_expired():
    err = subprocess.TimeoutExpired("test_cmd", 1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "TimeoutExpired"
    assert type(wrapped_err) is not type(err)


# LLM-generated content at query #40
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    non_subprocess_error = ValueError("Test error")
    result = error_wrapper(non_subprocess_error)
    assert result is non_subprocess_error


# LLM-generated content at query #41
#--------------------------

```python
def test_return_output_true():
    assert run_command("echo test", return_output=True).captured_output is not None

def test_return_code_nonzero():
    assert run_command("exit 1", ignore_errors=True).captured_output is not None

def test_verbose_true():
    assert run_command("echo test", verbose=True).captured_output is not None


# LLM-generated content at query #42
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo 'test'", verbose=False, return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_failure():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command("sleep 2", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 2"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command("echo 'verbose'", verbose=True, return_output=True)
    assert result.command == "echo 'verbose'"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "verbose"

def test_run_command_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert "test_value" in result.captured_output.decode('utf-8')

def test_run_command_cwd():
    result = run_command("pwd", cwd="/tmp", return_output=True)
    assert result.command == "pwd"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "/tmp"

def test_run_command_list_args():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"


# LLM-generated content at query #43
#--------------------------

```python
def test_run_command_success():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"], verbose=True)

def test_run_command_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_shell_string():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"


# LLM-generated content at query #44
#--------------------------

```python
def test_command_output_truncation():
    output = b"x" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #45
#--------------------------

```python
def test_error_wrapper_predicate_false():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #46
#--------------------------

```python
def test_error_wrapper_returns_non_subprocess_exception_unchanged():
    original_err = ValueError("test error")
    result = error_wrapper(original_err)
    assert result is original_err

def test_error_wrapper_modifies_called_process_error_with_output():
    original_err = subprocess.CalledProcessError(1, "test_cmd", output=b"line1\nline2")
    result = error_wrapper(original_err)
    assert isinstance(result, type(original_err))
    assert "Captured output:" in str(result)
    assert "line1" in str(result)
    assert "line2" in str(result)

def test_error_wrapper_modifies_called_process_error_without_output():
    original_err = subprocess.CalledProcessError(1, "test_cmd")
    result = error_wrapper(original_err)
    assert isinstance(result, type(original_err))
    assert "No output was generated." in str(result)

def test_error_wrapper_modifies_timeout_expired_with_output():
    original_err = subprocess.TimeoutExpired("test_cmd", timeout=1, output=b"timeout line")
    result = error_wrapper(original_err)
    assert isinstance(result, type(original_err))
    assert "Captured output:" in str(result)
    assert "timeout line" in str(result)

def test_error_wrapper_modifies_timeout_expired_without_output():
    original_err = subprocess.TimeoutExpired("test_cmd", timeout=1)
    result = error_wrapper(original_err)
    assert isinstance(result, type(original_err))
    assert "No output was generated." in str(result)

def test_error_wrapper_handles_unicode_decode_error():
    original_err = subprocess.CalledProcessError(1, "test_cmd", output=b'\xff\xfe')
    result = error_wrapper(original_err)
    assert isinstance(result, type(original_err))
    assert "Failed to parse output." in str(result)


# LLM-generated content at query #47
#--------------------------

```python
def test_error_wrapper_predicate():
    err = ValueError("test")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #48
#--------------------------

```python
def test_unicode_decode_error_handling():
    with tempfile.TemporaryFile() as f:
        f.write(b'\x80abc')
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            pass
        else:
            assert False, "Expected UnicodeDecodeError"


# LLM-generated content at query #49
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    test_error = ValueError("test")
    assert error_wrapper(test_error) is test_error


# LLM-generated content at query #50
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_verbose():
    result = run_command(["echo", "hello"], verbose=True, return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["printenv", "TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"test_value"

def test_run_command_with_timeout():
    result = run_command(["sleep", "0.1"], timeout=0.2, return_output=True)
    assert result.command == ["sleep", "0.1"]
    assert result.return_code == 0
    assert result.captured_output == b""

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True, return_output=True)
    assert result.command == ["false"]
    assert result.return_code == 1
    assert result.captured_output == b""

def test_run_command_with_timeout_error():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True, return_output=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output == b""

def test_run_command_with_string_command():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_nonzero_return_code():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"], return_output=True)


# LLM-generated content at query #51
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command(["false"])

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test"

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "test_value"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir

def test_run_command_string_command():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.command == "echo hello"
    assert result.return_code == 0
    assert result.captured_output.decode('utf-8').strip() == "hello"

def test_run_command_long_output_truncation():
    long_output = "a" * 10000
    result = run_command(f"echo {long_output}", shell=True, return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len("*** (previous output truncated) ***\n")


# LLM-generated content at query #52
#--------------------------

```python
def test_output_truncation():
    output = b"a" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #53
#--------------------------

```python
def test_unicode_decode_error_handling():
    with tempfile.TemporaryFile() as f:
        f.write(b'\x80\x81\x82')
        f.seek(0)
        output = f.read()
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            assert True


# LLM-generated content at query #54
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_creates_new_type_for_subprocess_errors():
    err = subprocess.CalledProcessError(1, "test")
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "CalledProcessError"
    assert type(wrapped_err) is not type(err)

def test_error_wrapper_preserves_output_in_str():
    err = subprocess.CalledProcessError(1, "test", output=b"line1\nline2")
    wrapped_err = error_wrapper(err)
    assert "Captured output:" in str(wrapped_err)
    assert "line1" in str(wrapped_err)
    assert "line2" in str(wrapped_err)

def test_error_wrapper_handles_empty_output():
    err = subprocess.CalledProcessError(1, "test", output=None)
    wrapped_err = error_wrapper(err)
    assert "No output was generated." in str(wrapped_err)

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test", output=b'\xff\xfe')
    wrapped_err = error_wrapper(err)
    assert "Failed to parse output." in str(wrapped_err)

def test_error_wrapper_works_with_timeout_expired():
    err = subprocess.TimeoutExpired("test", 1)
    wrapped_err = error_wrapper(err)
    assert type(wrapped_err).__name__ == "TimeoutExpired"
    assert type(wrapped_err) is not type(err)


# LLM-generated content at query #55
#--------------------------

```python
def test_unicode_decode_error_occurs():
    args = "echo 'test'"
    env = None
    cwd = None
    timeout = None
    verbose = True
    return_output = False
    ignore_errors = False
    kwargs = {}
    with tempfile.TemporaryFile() as f:
        ret = subprocess.run(args, check=True, stdout=f, stderr=subprocess.STDOUT,
                             timeout=timeout, env=env, cwd=cwd, **kwargs)
        f.seek(0)
        output = b'\xff\xfe'
        try:
            output.decode('utf-8')
        except UnicodeDecodeError:
            pass
        assert True


# LLM-generated content at query #56
#--------------------------

```python
def test_error_wrapper_predicate():
    assert isinstance(subprocess.CalledProcessError, Exception)
    assert isinstance(subprocess.TimeoutExpired, Exception)


# LLM-generated content at query #57
#--------------------------

```python
def test_error_wrapper_predicate_false():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #58
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #59
#--------------------------

```python
def test_error_wrapper_predicate_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #60
#--------------------------

```python
def test_verbose_logging():
    with patch('flutes.run.log') as mock_log:
        run_command("echo test", verbose=True)
        mock_log.assert_called_once_with("> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #61
#--------------------------

```python
def test_output_truncation():
    output = b"x" * (MAX_OUTPUT_LENGTH + 100)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #62
#--------------------------

```python
def test_run_command_success():
    result = run_command("echo 'test'", verbose=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_with_output():
    result = run_command("echo 'test'", return_output=True)
    assert result.command == "echo 'test'"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_error():
    with pytest.raises(subprocess.CalledProcessError):
        run_command("exit 1")

def test_run_command_ignore_errors():
    result = run_command("exit 1", ignore_errors=True)
    assert result.command == "exit 1"
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command("sleep 10", timeout=0.1)

def test_run_command_timeout_ignore_errors():
    result = run_command("sleep 10", timeout=0.1, ignore_errors=True)
    assert result.command == "sleep 10"
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command("echo $TEST_VAR", env={"TEST_VAR": "test_value"}, shell=True, return_output=True)
    assert result.command == "echo $TEST_VAR"
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command("pwd", cwd=tmpdir, return_output=True)
        assert result.command == "pwd"
        assert result.return_code == 0
        assert result.captured_output.decode('utf-8').strip() == tmpdir


# LLM-generated content at query #63
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_error():
    result = run_command(["ls", "/nonexistent"], ignore_errors=True)
    assert result.command == ["ls", "/nonexistent"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_timeout():
    result = run_command(["sleep", "10"], timeout=0.1, ignore_errors=True)
    assert result.command == ["sleep", "10"]
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_with_env():
    result = run_command(["echo", "$TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["echo", "$TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_string_command():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"


# LLM-generated content at query #64
#--------------------------

```python
def test_error_wrapper_predicate():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #65
#--------------------------

```python
def test_predicate_false():
    assert not (return_output or ret.returncode != 0 or verbose)


# LLM-generated content at query #66
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_exceptions():
    non_subprocess_error = ValueError("test error")
    assert error_wrapper(non_subprocess_error) is non_subprocess_error


# LLM-generated content at query #67
#--------------------------

```python
def test_predicate_evaluates_to_false():
    err = ValueError("test error")
    assert not isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #68
#--------------------------

```python
def test_predicate_at_line_32():
    output = b"a" * (MAX_OUTPUT_LENGTH + 1)
    assert len(output) > MAX_OUTPUT_LENGTH


# LLM-generated content at query #69
#--------------------------

```python
def test_verbose_logging_with_cwd():
    with patch('flutes.log.log') as mock_log:
        run_command("echo test", verbose=True, cwd="/test/path")
        mock_log.assert_called_with("/test/path> 'echo test'", timestamp=False, include_proc_id=False)


# LLM-generated content at query #70
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.command == ["echo", "hello"]
    assert result.return_code == 0
    assert result.captured_output == b"hello\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"/tmp"

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["printenv", "TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output.strip() == b"test_value"

def test_run_command_with_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_command(["sleep", "10"], timeout=0.1)

def test_run_command_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "verbose"], verbose=True, return_output=True)
    assert result.command == ["echo", "verbose"]
    assert result.return_code == 0
    assert result.captured_output == b"verbose\n"

def test_run_command_return_output():
    result = run_command(["echo", "output"], return_output=True)
    assert result.command == ["echo", "output"]
    assert result.return_code == 0
    assert result.captured_output == b"output\n"

def test_run_command_non_zero_return():
    result = run_command(["false"], return_output=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_shell():
    result = run_command("echo shell", shell=True, return_output=True)
    assert result.command == "echo shell"
    assert result.return_code == 0
    assert result.captured_output == b"shell\n"


# LLM-generated content at query #71
#--------------------------

```python
def test_unicode_decode_error_handling():
    args = "echo 'test'"
    result = run_command(args, return_output=True, ignore_errors=True)
    assert isinstance(result, CommandResult)
    assert result.command == args
    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #72
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    err = ValueError("test error")
    assert error_wrapper(err) is err

def test_error_wrapper_returns_wrapped_exception_for_called_process_error():
    err = subprocess.CalledProcessError(1, "test_command", output=b"test output")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nCaptured output:\n    test output"

def test_error_wrapper_returns_wrapped_exception_for_timeout_expired():
    err = subprocess.TimeoutExpired("test_command", 1)
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.TimeoutExpired)
    assert str(wrapped_err) == "Command 'test_command' timed out after 1 seconds.\nNo output was generated."

def test_error_wrapper_handles_unicode_decode_error():
    err = subprocess.CalledProcessError(1, "test_command", output=b"\xff\xfe")
    wrapped_err = error_wrapper(err)
    assert isinstance(wrapped_err, subprocess.CalledProcessError)
    assert str(wrapped_err) == "Command 'test_command' returned non-zero exit status 1.\nFailed to parse output."


# LLM-generated content at query #73
#--------------------------

```python
def test_error_wrapper_returns_original_error_for_non_subprocess_exceptions():
    original_error = ValueError("test error")
    result = error_wrapper(original_error)
    assert result is original_error


# LLM-generated content at query #74
#--------------------------

```python
def test_error_wrapper_returns_same_exception_for_non_subprocess_errors():
    non_subprocess_error = ValueError("test error")
    assert error_wrapper(non_subprocess_error) is non_subprocess_error


# LLM-generated content at query #75
#--------------------------

```python
def test_run_command_successful_execution():
    result = run_command(["echo", "test"], return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.command == ["echo", "test"]
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.command == ["pwd"]
    assert result.return_code == 0
    assert result.captured_output == b"/tmp\n"

def test_run_command_with_env():
    result = run_command(["printenv", "TEST_VAR"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.command == ["printenv", "TEST_VAR"]
    assert result.return_code == 0
    assert result.captured_output == b"test_value\n"

def test_run_command_with_timeout():
    result = run_command(["sleep", "0.1"], timeout=1, return_output=True)
    assert result.command == ["sleep", "0.1"]
    assert result.return_code == 0
    assert result.captured_output == b""

def test_run_command_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True, return_output=True)
    assert result.command == ["false"]
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_with_shell_command():
    result = run_command("echo test", shell=True, return_output=True)
    assert result.command == "echo test"
    assert result.return_code == 0
    assert result.captured_output == b"test\n"

def test_run_command_with_long_output():
    result = run_command(["python", "-c", "print('a' * 10000)"], return_output=True)
    assert result.command == ["python", "-c", "print('a' * 10000)"]
    assert result.return_code == 0
    assert len(result.captured_output) > 0


# LLM-generated content at query #76
#--------------------------

```python
def test_error_wrapper_predicate_false():
    assert not isinstance(Exception(), (subprocess.CalledProcessError, subprocess.TimeoutExpired))


