####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_error_wrapper_with_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_with_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial output')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial output' in result

def test_error_wrapper_with_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_other_exception():
    import subprocess
    err = ValueError('Some error')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #2
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "x" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_error_wrapper_includes_output():
    try:
        run_command(["python3", "-c", "import sys; sys.exit(1)"])
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)

def test_run_command_no_output_on_success():
    result = run_command(["true"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_nonzero_exit_with_output():
    result = run_command(["python3", "-c", "print('error'); exit(1)"], return_output=True)
    assert result.return_code == 1
    assert b"error" in result.captured_output


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command_predicate_true_when_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_predicate_true_when_returncode_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_predicate_true_when_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_predicate_true_when_return_output_true_and_verbose_true():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_predicate_true_when_returncode_nonzero_and_verbose_true():
    result = run_command(["false"], ignore_errors=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_predicate_true_when_return_output_true_and_returncode_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_predicate_true_when_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #4
#--------------------------

def test_error_wrapper_with_non_subprocess_exception():
    class CustomException(Exception):
        pass
    exc = CustomException("test")
    result = error_wrapper(exc)
    assert result is exc


# LLM-generated content at query #5
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_no_ignore():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output is not None

def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout_no_ignore():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"/bin/bash" in result.captured_output or b"/bin/sh" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" not in result.captured_output
    result = run_command(["python3", "-c", f"import sys; sys.stdout.write('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) <= 8192 + len(b"*** (previous output truncated) ***\n")

def test_run_command_nonzero_return_with_output():
    result = run_command(["python3", "-c", "import sys; print('error'); sys.exit(1)"], return_output=False)
    assert result.return_code == 1
    assert result.captured_output is not None
    assert b"error" in result.captured_output


# LLM-generated content at query #6
#--------------------------

```python
def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None
    assert result.command == ["echo", "hello"]

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output
    assert result.command == ["echo", "hello"]

def test_run_command_failure_raises_exception():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert b"No output was generated" in str(e).encode() or e.output is not None

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None
    assert result.command == ["false"]

def test_run_command_timeout_raises_exception():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1
        assert b"Captured output" in str(e).encode() or e.output is not None

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None
    assert result.command == ["sleep", "2"]

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "test_value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=test_value" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output
    assert result.command == "echo hello"

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_nonzero_exit_with_output():
    result = run_command(["bash", "-c", "echo error; exit 1"], return_output=False)
    assert result.return_code == 1
    assert result.captured_output is not None
    assert b"error" in result.captured_output

def test_run_command_truncates_long_output():
    long_output = "x" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_unicode_output():
    result = run_command(["printf", "你好"], return_output=True)
    assert result.return_code == 0
    assert "你好".encode() in result.captured_output


# LLM-generated content at query #7
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired:
        pass

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0

def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "a" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_error_wrapper_output():
    try:
        run_command(["python3", "-c", "import sys; print('error'); sys.exit(1)"])
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)

def test_run_command_no_output_on_success():
    result = run_command(["true"])
    assert result.captured_output is None

def test_run_command_output_on_nonzero_return():
    result = run_command(["false"], return_output=True)
    assert result.captured_output is not None


# LLM-generated content at query #8
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_error_ignored():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignored():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "a" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_nonzero_exit_no_output():
    result = run_command(["false"], return_output=False)
    assert result.return_code != 0
    assert result.captured_output is not None


# LLM-generated content at query #9
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.CalledProcessError.__str__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1)
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.TimeoutExpired.__str__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError("test")
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert wrapped.__class__.__str__ == ValueError.__str__

def test_error_wrapper_preserves_exception_attributes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=2, cmd=['cat'], output=b'output')
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 2
    assert wrapped.cmd == ['cat']
    assert wrapped.output == b'output'

def test_error_wrapper_str_includes_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['echo'], output=b'hello\nworld')
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Captured output:' in str_repr
    assert '    hello' in str_repr
    assert '    world' in str_repr

def test_error_wrapper_str_includes_no_output_message():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'No output was generated.' in str_repr

def test_error_wrapper_str_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['cat'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Failed to parse output.' in str_repr


# LLM-generated content at query #10
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_verbose():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true_and_nonzero():
    result = run_command(["false"], verbose=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #11
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


# LLM-generated content at query #12
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_with_output():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0

def test_run_command_with_cwd():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_shell_false():
    result = run_command(["echo", "hello"], shell=False, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_unicode_output():
    result = run_command(["printf", "hello\\x80world"], return_output=True)
    assert result.return_code == 0

def test_run_command_error_not_ignored():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_timeout_not_ignored():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_error_wrapper_str():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        str_output = str(e)
        assert "Captured output:" in str_output or "No output was generated." in str_output

def test_run_command_timeout_wrapper_str():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        str_output = str(e)
        assert "Captured output:" in str_output or "No output was generated." in str_output


# LLM-generated content at query #13
#--------------------------

```python
def test_run_command_verbose_logging_without_cwd():
    result = run_command(["echo", "test"], verbose=True, cwd=None)
    assert result.return_code == 0

def test_run_command_verbose_logging_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["echo", "test"], verbose=True, cwd=tmpdir)
        assert result.return_code == 0

def test_run_command_verbose_logging_with_shell_command():
    result = run_command("echo test", verbose=True, shell=True)
    assert result.return_code == 0

def test_run_command_verbose_logging_with_list_args():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0

def test_run_command_verbose_logging_with_env_vars():
    result = run_command(["echo", "test"], verbose=True, env={"TEST_VAR": "value"})
    assert result.return_code == 0


# LLM-generated content at query #14
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output


def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output


def test_run_command_output_truncation():
    long_output = "x" * 10000
    result = run_command(["printf", long_output], return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) < 10000
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #15
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_error_raise():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_timeout_raise():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["echo", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #16
#--------------------------

def test_run_command_output_truncated_when_exceeds_max_length():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    MAX_OUTPUT_LENGTH = 8192
    long_output = b"x" * (MAX_OUTPUT_LENGTH + 100)
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.stdout = None
    mock_process.stderr = None
    with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd', output=long_output)):
        with tempfile.TemporaryFile() as mock_file:
            mock_file.write = MagicMock()
            mock_file.seek = MagicMock()
            mock_file.read = MagicMock(return_value=long_output)
            with patch('tempfile.TemporaryFile', return_value=mock_file):
                from flutes.run import run_command
                result = run_command('cmd', ignore_errors=True)
                assert result.captured_output is not None
                assert len(result.captured_output) == len(b"*** (previous output truncated) ***\n") + MAX_OUTPUT_LENGTH
                assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
                assert result.captured_output.endswith(b"x" * MAX_OUTPUT_LENGTH)


# LLM-generated content at query #17
#--------------------------

```python
def test_run_command_no_output_on_success():
    result = run_command(["echo", "test"], return_output=False, verbose=False)
    assert result.captured_output is None


# LLM-generated content at query #18
#--------------------------

```python
def test_run_command_verbose_with_unicode_decode_error():
    import subprocess
    import tempfile
    from flutes.run import run_command
    from flutes.log import log
    import sys
    import os

    def mock_subprocess_run(args, **kwargs):
        class MockCompletedProcess:
            returncode = 0
        return MockCompletedProcess()

    def mock_log(msg, **kwargs):
        pass

    original_subprocess_run = subprocess.run
    original_log = log
    subprocess.run = mock_subprocess_run
    log = mock_log

    try:
        with tempfile.TemporaryFile() as f:
            f.write(b'\xff\xfe\xfd')
            f.seek(0)
            result = run_command(["echo", "test"], verbose=True, return_output=True)
    finally:
        subprocess.run = original_subprocess_run
        log = original_log

    assert result.return_code == 0
    assert result.captured_output is not None


# LLM-generated content at query #19
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired:
        pass

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "a" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_error_wrapper_includes_output():
    try:
        run_command(["python3", "-c", "import sys; print('error'); sys.exit(1)"])
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)
        assert "error" in str(e)


# LLM-generated content at query #20
#--------------------------

```python
def test_run_command_verbose_logs_without_timestamp_and_proc_id():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock, call
    from flutes.run import run_command
    from flutes.log import log

    with patch('subprocess.run') as mock_run, patch('flutes.run.log') as mock_log:
        mock_run.return_value = MagicMock(returncode=0)
        args = ["echo", "hello"]
        run_command(args, verbose=True)
        mock_log.assert_called_once_with('> ["echo", "hello"]', timestamp=False, include_proc_id=False)


# LLM-generated content at query #21
#--------------------------

```python
def test_run_command_no_output_on_success():
    result = run_command(["echo", "test"], return_output=False)
    assert result.captured_output is None


# LLM-generated content at query #22
#--------------------------

```python
def test_run_command_output_truncation():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    mock_subprocess_run = MagicMock(side_effect=subprocess.CalledProcessError(1, 'cmd'))
    with patch('subprocess.run', mock_subprocess_run):
        with tempfile.TemporaryFile() as mock_file:
            mock_file.write = MagicMock()
            mock_file.seek = MagicMock()
            mock_file.read = MagicMock(return_value=b'a' * 10000)
            with patch('tempfile.TemporaryFile', return_value=mock_file):
                result = run_command(['echo', 'test'], ignore_errors=True)
    assert len(result.captured_output) < 10000
    assert b'*** (previous output truncated) ***' in result.captured_output


# LLM-generated content at query #23
#--------------------------

def test_error_wrapper_returns_err_when_not_subprocess_error():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #24
#--------------------------

def test_error_wrapper_returns_err_when_not_subprocess_error():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #25
#--------------------------

```python
def test_run_command_verbose_with_unicode_decode_error():
    import subprocess
    import tempfile
    from flutes.run import run_command
    from unittest.mock import patch, MagicMock

    with patch('subprocess.run') as mock_run:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        with patch('flutes.run.log') as mock_log:
            with tempfile.TemporaryFile() as mock_file:
                with patch('tempfile.TemporaryFile', return_value=mock_file):
                    mock_file.seek = MagicMock()
                    mock_file.read = MagicMock(return_value=b'\xff\xfe\xfd')
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock()
                    
                    result = run_command(["echo", "test"], verbose=True, return_output=False)
                    
                    assert mock_log.call_count >= 2
                    split_line_call_found = False
                    for call in mock_log.call_args_list:
                        args, kwargs = call
                        if args[0] == str(b'\xff\xfe\xfd') or args[0] == str(b''):
                            split_line_call_found = True
                            break
                    assert split_line_call_found


# LLM-generated content at query #26
#--------------------------

```python
def test_run_command_output_truncation_when_exceeding_max_length():
    import subprocess
    import tempfile
    from flutes.run import run_command
    from flutes.run import MAX_OUTPUT_LENGTH
    long_output = b"x" * (MAX_OUTPUT_LENGTH + 100)
    with tempfile.NamedTemporaryFile(mode='wb') as tmp:
        tmp.write(long_output)
        tmp.flush()
        result = run_command(["cat", tmp.name], shell=False, ignore_errors=True)
    assert result.captured_output is not None
    assert len(result.captured_output) < len(long_output)
    assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
    assert result.captured_output.endswith(b"x" * MAX_OUTPUT_LENGTH)


# LLM-generated content at query #27
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.CalledProcessError.__str__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.TimeoutExpired.__str__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert wrapped.__class__ is ValueError

def test_wrapped_called_process_error_str_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_wrapped_called_process_error_str_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_wrapped_called_process_error_str_with_unicode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls', output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result

def test_wrapped_timeout_expired_str_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1, output=b'output line')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    output line' in result

def test_wrapped_timeout_expired_str_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result


# LLM-generated content at query #28
#--------------------------

def test_error_wrapper_returns_err_when_not_subprocess_error():
    import subprocess
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #29
#--------------------------

def test_error_wrapper_with_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_with_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_with_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert str(wrapped) == 'test'

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #30
#--------------------------

def test_error_wrapper_with_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_with_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_with_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #31
#--------------------------

def test_error_wrapper_with_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_with_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial output')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial output' in result

def test_error_wrapper_with_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_other_exception():
    import subprocess
    err = ValueError('Some error')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['cmd'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #32
#--------------------------

def test_error_wrapper_non_subprocess_exception():
    class CustomException(Exception):
        pass
    exc = CustomException("test")
    result = error_wrapper(exc)
    assert result is exc


# LLM-generated content at query #33
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #34
#--------------------------

def test_error_wrapper_with_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_with_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_with_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_with_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['cmd'], output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_return_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_error_with_exception():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output is not None


def test_run_command_error_with_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None


def test_run_command_timeout_with_exception():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1
        assert e.output is not None


def test_run_command_timeout_with_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output


def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_shell_true():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"/bin/bash" in result.captured_output or b"/bin/sh" in result.captured_output


# LLM-generated content at query #2
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None


def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output


def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None


def test_run_command_error_raise():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None


def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None


def test_run_command_timeout_raise():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None


def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None


def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert b"TEST_VAR=value" in result.captured_output


def test_run_command_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert b"/tmp" in result.captured_output


def test_run_command_shell():
    result = run_command("echo hello", shell=True, return_output=True)
    assert b"hello" in result.captured_output


def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #3
#--------------------------

```python
def test_run_command_no_output_on_success():
    result = run_command(["echo", "test"], return_output=False, verbose=False)
    assert result.captured_output is None


# LLM-generated content at query #4
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_no_ignore():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_no_ignore():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo $HOME", shell=True, return_output=True)
    assert result.return_code == 0
    assert len(result.captured_output) > 0

def test_run_command_output_truncation():
    long_output = "x" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #5
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None
    assert b"hello" in result.captured_output

def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode != 0
        assert e.output is not None

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code != 0
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert result.return_code == 0
        assert tmpdir.encode() in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_error_wrapper_output():
    try:
        run_command(["python3", "-c", "import sys; sys.exit(1)"])
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)

def test_run_command_no_output_on_success():
    result = run_command(["true"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_nonzero_exit_with_output():
    result = run_command(["python3", "-c", "print('error'); exit(1)"], return_output=True)
    assert result.return_code == 1
    assert b"error" in result.captured_output


# LLM-generated content at query #6
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output is not None

def test_run_command_error_ignored():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1
        assert e.output is not None

def test_run_command_timeout_ignored():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_output_truncation():
    long_output = "x" * 10000
    result = run_command(["printf", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_nonzero_return_with_output():
    result = run_command(["sh", "-c", "echo error; exit 1"], return_output=True)
    assert result.return_code == 1
    assert b"error" in result.captured_output


# LLM-generated content at query #7
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_verbose():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true_and_nonzero():
    result = run_command(["false"], verbose=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #8
#--------------------------

```python
def test_run_command_verbose_logs_without_timestamp_and_proc_id():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command
    from flutes.log import log

    with patch('subprocess.run') as mock_run, patch('flutes.run.log') as mock_log:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        args = ['echo', 'test']
        result = run_command(args, verbose=True)
        mock_log.assert_called_once_with("> ['echo', 'test']", timestamp=False, include_proc_id=False)
        assert result.return_code == 0
        assert result.captured_output is None


# LLM-generated content at query #9
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['cmd'], output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #10
#--------------------------

def test_error_wrapper_non_subprocess_exception():
    class CustomException(Exception):
        pass
    exc = CustomException("test")
    result = error_wrapper(exc)
    assert result is exc


# LLM-generated content at query #11
#--------------------------

```python
def test_run_command_verbose_logs_without_timestamp_and_proc_id():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command
    from flutes.log import log

    with patch('subprocess.run') as mock_run, patch('flutes.run.log') as mock_log:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        run_command(["echo", "test"], verbose=True)
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[1]['timestamp'] == False
        assert call_args[1]['include_proc_id'] == False


# LLM-generated content at query #12
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_called_process_error_empty_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial output')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial output' in result

def test_error_wrapper_timeout_expired_no_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('test error')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_called_process_error_non_utf8_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['cat'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result

def test_error_wrapper_called_process_error_output_with_newlines():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['echo'], output=b'line1\nline2\nline3')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert '    line1' in result
    assert '    line2' in result
    assert '    line3' in result

def test_error_wrapper_called_process_error_output_trailing_newline():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['echo'], output=b'line1\nline2\n')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert '    line1' in result
    assert '    line2' in result
    assert result.count('\n') >= 3

def test_error_wrapper_timeout_expired_output_with_tabs():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'\tindented')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert '    \tindented' in result


# LLM-generated content at query #13
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_no_ignore():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output is not None

def test_run_command_error_ignore():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout_no_ignore():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose(capsys):
    run_command(["echo", "test"], verbose=True)
    captured = capsys.readouterr()
    assert "test" in captured.out

def test_run_command_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert b"/tmp" in result.captured_output

def test_run_command_shell():
    result = run_command("echo hello", shell=True, return_output=True)
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_error_wrapper_str():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        str_repr = str(e)
        assert "Captured output:" in str_repr

def test_run_command_no_output_on_success():
    result = run_command(["true"])
    assert result.captured_output is None

def test_run_command_with_kwargs():
    result = run_command(["echo", "hello"], stdin=subprocess.DEVNULL, return_output=True)
    assert b"hello" in result.captured_output


# LLM-generated content at query #14
#--------------------------

```python
def test_run_command_return_output_true():
    result = run_command(["echo", "test"], return_output=True)
    assert result.captured_output is not None

def test_run_command_return_code_nonzero():
    result = run_command(["false"], ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true():
    result = run_command(["echo", "test"], verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_verbose_true():
    result = run_command(["echo", "test"], return_output=True, verbose=True)
    assert result.captured_output is not None

def test_run_command_return_output_true_and_return_code_nonzero():
    result = run_command(["false"], return_output=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_verbose_true_and_return_code_nonzero():
    result = run_command(["false"], verbose=True, ignore_errors=True)
    assert result.captured_output is not None

def test_run_command_all_conditions_true():
    result = run_command(["false"], return_output=True, verbose=True, ignore_errors=True)
    assert result.captured_output is not None


# LLM-generated content at query #15
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output is not None

def test_run_command_error_ignored():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1
        assert e.output is not None

def test_run_command_timeout_ignored():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    env = {"TEST_VAR": "value"}
    result = run_command(["env"], env=env, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo $SHELL", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"/bin/bash" in result.captured_output or b"/bin/sh" in result.captured_output

def test_run_command_output_truncation():
    long_output = "x" * 10000
    result = run_command(["printf", long_output], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output


# LLM-generated content at query #16
#--------------------------

```python
def test_run_command_verbose_with_unicode_decode_error():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with patch('flutes.run.log') as mock_log:
            mock_log.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
            result = run_command(['echo', 'test'], verbose=True, return_output=True)
            assert mock_log.call_count > 0
            assert isinstance(result.captured_output, bytes)


# LLM-generated content at query #17
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = b'output line 1\noutput line 2'
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert '__str__' in wrapped.__class__.__dict__
    str_output = str(wrapped)
    assert 'Captured output:' in str_output
    assert '    output line 1' in str_output
    assert '    output line 2' in str_output

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1)
    err.output = b'timeout output'
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert '__str__' in wrapped.__class__.__dict__
    str_output = str(wrapped)
    assert 'Captured output:' in str_output
    assert '    timeout output' in str_output

def test_error_wrapper_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['echo'])
    err.output = b'\xff\xfe'
    wrapped = error_wrapper(err)
    assert wrapped is err
    str_output = str(wrapped)
    assert 'Failed to parse output.' in str_output

def test_error_wrapper_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['false'])
    err.output = None
    wrapped = error_wrapper(err)
    assert wrapped is err
    str_output = str(wrapped)
    assert 'No output was generated.' in str_output

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test error')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert not isinstance(wrapped, (subprocess.CalledProcessError, subprocess.TimeoutExpired))


# LLM-generated content at query #18
#--------------------------

```python
def test_run_command_output_truncation_when_exceeding_max_length():
    import subprocess
    import tempfile
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    mock_subprocess_run = MagicMock()
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, 'cmd')
    with patch('subprocess.run', mock_subprocess_run):
        with patch('flutes.run.MAX_OUTPUT_LENGTH', 8192):
            with tempfile.TemporaryFile() as mock_file:
                mock_file.write(b'a' * 10000)
                mock_file.seek = MagicMock()
                mock_file.read = MagicMock(return_value=b'a' * 10000)
                with patch('tempfile.TemporaryFile', return_value=mock_file):
                    result = run_command(['test'], ignore_errors=True)
                    assert result.captured_output.startswith(b'*** (previous output truncated) ***\n')
                    assert len(result.captured_output) == 8192 + len(b'*** (previous output truncated) ***\n')


# LLM-generated content at query #19
#--------------------------

```python
def test_run_command_output_truncation_when_exceeding_max_length():
    import subprocess
    import tempfile
    import sys
    from unittest.mock import patch, MagicMock
    from flutes.run import run_command

    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, 'cmd')
        with patch('tempfile.TemporaryFile') as mock_tempfile:
            mock_file = MagicMock()
            mock_file.seek = MagicMock()
            mock_file.read = MagicMock(return_value=b'a' * 9000)
            mock_tempfile.return_value.__enter__.return_value = mock_file
            result = run_command(['test'], ignore_errors=True)
            assert len(result.captured_output) > 8192
            assert result.captured_output.startswith(b'*** (previous output truncated) ***\n')
            assert len(result.captured_output) == len(b'*** (previous output truncated) ***\n') + 8192


# LLM-generated content at query #20
#--------------------------

```python
def test_run_command_output_truncation_when_exceeding_max_length():
    MAX_OUTPUT_LENGTH = 8192
    long_output = b"x" * (MAX_OUTPUT_LENGTH + 100)
    mock_subprocess_run = lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "cmd"))
    original_subprocess_run = subprocess.run
    subprocess.run = mock_subprocess_run
    try:
        result = run_command("echo test", ignore_errors=True)
        assert len(result.captured_output) <= MAX_OUTPUT_LENGTH + len(b"*** (previous output truncated) ***\n")
        assert result.captured_output.startswith(b"*** (previous output truncated) ***\n")
    finally:
        subprocess.run = original_subprocess_run


# LLM-generated content at query #21
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_timeout_expired_no_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError("test")
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_called_process_error_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #22
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    err.output = b'some output'
    wrapped = error_wrapper(err)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert hasattr(wrapped.__class__, '__str__')
    assert '__str__' in wrapped.__class__.__dict__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1)
    err.output = b'timeout output'
    wrapped = error_wrapper(err)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert hasattr(wrapped.__class__, '__str__')
    assert '__str__' in wrapped.__class__.__dict__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('some error')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)

def test_error_wrapper_preserves_exception_attributes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=2, cmd=['cat', 'file'])
    err.output = b'file not found'
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 2
    assert wrapped.cmd == ['cat', 'file']
    assert wrapped.output == b'file not found'

def test_error_wrapper_new_type_has_correct_str_method():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['echo'])
    err.output = b'hello\nworld'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Captured output:' in str_repr
    assert '    hello' in str_repr
    assert '    world' in str_repr

def test_error_wrapper_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['echo'])
    err.output = b'\xff\xfe'
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'Failed to parse output.' in str_repr

def test_error_wrapper_handles_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['echo'])
    err.output = None
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert 'No output was generated.' in str_repr

def test_error_wrapper_original_class_not_modified():
    import subprocess
    original_str = subprocess.CalledProcessError.__str__
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'])
    wrapped = error_wrapper(err)
    assert subprocess.CalledProcessError.__str__ is original_str


# LLM-generated content at query #23
#--------------------------

def test_error_wrapper_wraps_subprocess_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='ls')
    err.output = b'file1\nfile2'
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert '__str__' in wrapped.__class__.__dict__
    str_output = str(wrapped)
    assert 'Captured output:' in str_output
    assert '    file1' in str_output
    assert '    file2' in str_output

def test_error_wrapper_wraps_subprocess_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired(cmd='sleep 10', timeout=1)
    err.output = b'timeout output'
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, type(err))
    assert '__str__' in wrapped.__class__.__dict__
    str_output = str(wrapped)
    assert 'Captured output:' in str_output
    assert '    timeout output' in str_output

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert type(wrapped) is ValueError

def test_error_wrapper_handles_unicode_decode_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='cmd')
    err.output = b'\xff\xfe'
    wrapped = error_wrapper(err)
    str_output = str(wrapped)
    assert 'Failed to parse output.' in str_output

def test_error_wrapper_with_no_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd='cmd')
    err.output = None
    wrapped = error_wrapper(err)
    str_output = str(wrapped)
    assert 'No output was generated.' in str_output

def test_error_wrapper_preserves_exception_attributes():
    import subprocess
    err = subprocess.CalledProcessError(returncode=5, cmd='test_cmd')
    err.output = b'output'
    wrapped = error_wrapper(err)
    assert wrapped.returncode == 5
    assert wrapped.cmd == 'test_cmd'
    assert wrapped.output == b'output'


# LLM-generated content at query #24
#--------------------------

def test_error_wrapper_returns_err_when_not_subprocess_error():
    err = ValueError("test error")
    result = error_wrapper(err)
    assert result is err


# LLM-generated content at query #25
#--------------------------

```python
def test_run_command_no_output_on_success():
    result = run_command(["echo", "test"], return_output=False, verbose=False)
    assert result.captured_output is None
    assert result.return_code == 0


# LLM-generated content at query #26
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_failure_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output is not None

def test_run_command_failure_ignore_errors():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.output is not None

def test_run_command_timeout_ignore_errors():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True)
    assert result.return_code == 0
    assert result.captured_output is not None

def test_run_command_with_cwd():
    result = run_command(["pwd"], cwd="/tmp", return_output=True)
    assert result.return_code == 0
    assert b"/tmp" in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert result.return_code == 0
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "x" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert result.return_code == 0
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_error_wrapper_output():
    try:
        run_command(["python3", "-c", "import sys; sys.exit(1)"])
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)

def test_run_command_no_output_on_success():
    result = run_command(["true"])
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_nonzero_exit_with_output():
    result = run_command(["python3", "-c", "print('error'); exit(1)"], return_output=True)
    assert result.return_code == 1
    assert b"error" in result.captured_output


# LLM-generated content at query #27
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError("test error")
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #28
#--------------------------

def test_error_wrapper_subprocess_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_subprocess_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_subprocess_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial' in result

def test_error_wrapper_subprocess_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    err = ValueError("test")
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #29
#--------------------------

```python
def test_run_command_no_output_on_success():
    result = run_command(["echo", "test"], return_output=False, verbose=False)
    assert result.captured_output is None


# LLM-generated content at query #30
#--------------------------

```python
def test_run_command_verbose_with_non_utf8_output():
    import subprocess
    import tempfile
    from flutes.run import run_command
    from flutes.log import log
    import sys
    import io

    # Capture log output
    log_capture = io.StringIO()
    original_log_fn = log.__globals__['_CONSOLE_LOG_FN']
    log.__globals__['_CONSOLE_LOG_FN'] = lambda msg: log_capture.write(msg + '\n')

    # Create a command that outputs non-UTF-8 bytes
    test_output = b'\xff\xfe\xfd\xfc'
    args = [sys.executable, '-c', f'import sys; sys.stdout.buffer.write({test_output!r})']

    result = run_command(args, verbose=True, return_output=True)

    # Restore original log function
    log.__globals__['_CONSOLE_LOG_FN'] = original_log_fn

    # Check that the predicate at line 46 evaluated to True (UnicodeDecodeError occurred)
    # by verifying that the output was logged line by line as strings
    log_output = log_capture.getvalue()
    assert str(test_output) in log_output
    assert result.captured_output == test_output


# LLM-generated content at query #31
#--------------------------

def test_error_wrapper_non_subprocess_exception():
    class CustomException(Exception):
        pass
    exc = CustomException("test")
    result = error_wrapper(exc)
    assert result is exc
    assert not isinstance(result, (subprocess.CalledProcessError, subprocess.TimeoutExpired))

def test_error_wrapper_called_process_error_no_output():
    exc = subprocess.CalledProcessError(1, ["cmd"])
    exc.output = None
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.output is None
    assert "No output was generated." in str(result)

def test_error_wrapper_called_process_error_with_output():
    exc = subprocess.CalledProcessError(1, ["cmd"])
    exc.output = b"test output"
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.output == b"test output"
    assert "Captured output:" in str(result)
    assert "    test output" in str(result)

def test_error_wrapper_timeout_expired_no_output():
    exc = subprocess.TimeoutExpired(["cmd"], timeout=10)
    exc.output = None
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.TimeoutExpired)
    assert result.output is None
    assert "No output was generated." in str(result)

def test_error_wrapper_timeout_expired_with_output():
    exc = subprocess.TimeoutExpired(["cmd"], timeout=10)
    exc.output = b"partial output"
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.TimeoutExpired)
    assert result.output == b"partial output"
    assert "Captured output:" in str(result)
    assert "    partial output" in str(result)

def test_error_wrapper_unicode_decode_error():
    exc = subprocess.CalledProcessError(1, ["cmd"])
    exc.output = b"\xff\xfe"
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.output == b"\xff\xfe"
    assert "Failed to parse output." in str(result)

def test_error_wrapper_multiline_output():
    exc = subprocess.CalledProcessError(1, ["cmd"])
    exc.output = b"line1\nline2\nline3"
    result = error_wrapper(exc)
    assert isinstance(result, subprocess.CalledProcessError)
    assert result.output == b"line1\nline2\nline3"
    str_result = str(result)
    assert "    line1" in str_result
    assert "    line2" in str_result
    assert "    line3" in str_result


# LLM-generated content at query #32
#--------------------------

def test_error_wrapper_returns_same_error_for_non_subprocess_exceptions():
    class CustomException(Exception):
        pass
    custom_err = CustomException("test")
    result = error_wrapper(custom_err)
    assert result is custom_err
    assert isinstance(result, CustomException)


# LLM-generated content at query #33
#--------------------------

def test_error_wrapper_non_subprocess_exception():
    class CustomException(Exception):
        pass
    exc = CustomException("test")
    result = error_wrapper(exc)
    assert result is exc


# LLM-generated content at query #34
#--------------------------

def test_error_wrapper_called_process_error_with_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'file1\nfile2')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    file1' in result
    assert '    file2' in result

def test_error_wrapper_called_process_error_without_output():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_timeout_expired_with_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=b'partial output')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Captured output:' in result
    assert '    partial output' in result

def test_error_wrapper_timeout_expired_without_output():
    import subprocess
    err = subprocess.TimeoutExpired(cmd=['sleep', '10'], timeout=1, output=None)
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'No output was generated.' in result

def test_error_wrapper_other_exception():
    import subprocess
    err = ValueError('Some error')
    wrapped = error_wrapper(err)
    assert wrapped is err

def test_error_wrapper_output_decoding_error():
    import subprocess
    err = subprocess.CalledProcessError(returncode=1, cmd=['ls'], output=b'\xff\xfe')
    wrapped = error_wrapper(err)
    result = str(wrapped)
    assert 'Failed to parse output.' in result


# LLM-generated content at query #35
#--------------------------

def test_error_wrapper_wraps_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert wrapped.__class__.__name__ == 'CalledProcessError'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.CalledProcessError.__str__

def test_error_wrapper_wraps_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 10)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert wrapped.__class__.__name__ == 'TimeoutExpired'
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ != subprocess.TimeoutExpired.__str__

def test_error_wrapper_returns_other_exceptions_unchanged():
    import subprocess
    err = ValueError('test')
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert wrapped.__class__ is ValueError

def test_error_wrapper_predicate_true_for_called_process_error():
    import subprocess
    err = subprocess.CalledProcessError(1, 'cmd')
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result is True

def test_error_wrapper_predicate_true_for_timeout_expired():
    import subprocess
    err = subprocess.TimeoutExpired('cmd', 10)
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result is True

def test_error_wrapper_predicate_false_for_other_exception():
    import subprocess
    err = KeyError('test')
    result = isinstance(err, (subprocess.CalledProcessError, subprocess.TimeoutExpired))
    assert result is False


# LLM-generated content at query #36
#--------------------------

def test_error_wrapper_wraps_subprocess_exceptions():
    import subprocess
    from typing import Type
    ExcType = Type[Exception]
    result = subprocess.run(["false"], capture_output=True)
    err = subprocess.CalledProcessError(result.returncode, ["false"], output=result.stdout, stderr=result.stderr)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.CalledProcessError)
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ is not subprocess.CalledProcessError.__str__

def test_error_wrapper_does_not_wrap_other_exceptions():
    import subprocess
    from typing import Type
    ExcType = Type[Exception]
    err = ValueError("test error")
    wrapped = error_wrapper(err)
    assert wrapped is err
    assert isinstance(wrapped, ValueError)
    assert not isinstance(wrapped, (subprocess.CalledProcessError, subprocess.TimeoutExpired))

def test_error_wrapper_with_timeout_expired():
    import subprocess
    from typing import Type
    ExcType = Type[Exception]
    err = subprocess.TimeoutExpired(["sleep", "10"], timeout=1)
    wrapped = error_wrapper(err)
    assert isinstance(wrapped, subprocess.TimeoutExpired)
    assert hasattr(wrapped.__class__, '__str__')
    assert wrapped.__class__.__str__ is not subprocess.TimeoutExpired.__str__

def test_wrapped_exception_str_with_output():
    import subprocess
    from typing import Type
    ExcType = Type[Exception]
    output = b"Hello\nWorld"
    err = subprocess.CalledProcessError(1, ["cmd"], output=output)
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "Captured output:" in str_repr
    assert "    Hello" in str_repr
    assert "    World" in str_repr

def test_wrapped_exception_str_without_output():
    import subprocess
    from typing import Type
    ExcType = Type[Exception]
    err = subprocess.CalledProcessError(1, ["cmd"], output=None)
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "No output was generated." in str_repr

def test_wrapped_exception_str_with_unicode_error():
    import subprocess
    from typing import Type
    ExcType = Type[Exception]
    output = b'\xff\xfe'
    err = subprocess.CalledProcessError(1, ["cmd"], output=output)
    wrapped = error_wrapper(err)
    str_repr = str(wrapped)
    assert "Failed to parse output." in str_repr


# LLM-generated content at query #37
#--------------------------

def test_run_command_success_no_output():
    result = run_command(["echo", "hello"], return_output=False)
    assert result.return_code == 0
    assert result.captured_output is None

def test_run_command_success_with_output():
    result = run_command(["echo", "hello"], return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_error_raises():
    try:
        run_command(["false"])
    except subprocess.CalledProcessError as e:
        assert e.returncode == 1
        assert e.output is not None

def test_run_command_error_ignored():
    result = run_command(["false"], ignore_errors=True)
    assert result.return_code == 1
    assert result.captured_output is not None

def test_run_command_timeout_raises():
    try:
        run_command(["sleep", "2"], timeout=0.1)
    except subprocess.TimeoutExpired as e:
        assert e.timeout == 0.1
        assert e.output is not None

def test_run_command_timeout_ignored():
    result = run_command(["sleep", "2"], timeout=0.1, ignore_errors=True)
    assert result.return_code == -32768
    assert result.captured_output is not None

def test_run_command_verbose():
    result = run_command(["echo", "test"], verbose=True, return_output=True)
    assert result.return_code == 0
    assert b"test" in result.captured_output

def test_run_command_with_cwd():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_command(["pwd"], cwd=tmpdir, return_output=True)
        assert tmpdir.encode() in result.captured_output

def test_run_command_with_env():
    result = run_command(["env"], env={"TEST_VAR": "value"}, return_output=True)
    assert b"TEST_VAR=value" in result.captured_output

def test_run_command_shell_true():
    result = run_command("echo hello", shell=True, return_output=True)
    assert result.return_code == 0
    assert b"hello" in result.captured_output

def test_run_command_output_truncation():
    long_output = "A" * 10000
    result = run_command(["python3", "-c", f"print('{long_output}')"], return_output=True)
    assert b"*** (previous output truncated) ***" in result.captured_output

def test_run_command_error_wrapper_output():
    try:
        run_command(["python3", "-c", "import sys; print('error'); sys.exit(1)"])
    except subprocess.CalledProcessError as e:
        assert "Captured output:" in str(e)
        assert "error" in str(e)


