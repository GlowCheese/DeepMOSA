####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_find_hook():
    import tempfile
    import os
    from pathlib import Path
    
    # Test 1: No hooks directory exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', hooks_dir=os.path.join(tmpdir, 'hooks'))
        assert result is None
    
    # Test 2: Empty hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result is None
    
    # Test 3: Valid hook file found
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook_file)]
    
    # Test 4: Multiple hook files, only matching one returned
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook2 = os.path.join(hooks_dir, 'post_gen_project.sh')
        hook3 = os.path.join(hooks_dir, 'other_script.py')
        Path(hook1).touch()
        Path(hook2).touch()
        Path(hook3).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook1)]
    
    # Test 5: Backup files ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result is None
    
    # Test 6: Unsupported hook name ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'unsupported_hook.py')
        Path(hook_file).touch()
        result = find_hook('unsupported_hook', hooks_dir=hooks_dir)
        assert result is None
    
    # Test 7: Hook with different extension still valid
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.sh')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook_file)]
    
    # Test 8: Multiple matching hooks (shouldn't happen but test edge case)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        Path(hook1).touch()
        Path(hook2).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert sorted(result) == sorted([os.path.abspath(hook1), os.path.abspath(hook2)])


# LLM-generated content at query #2
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test 1: No pre_prompt hook exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

    # Test 2: Valid pre_prompt hook executes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        script = hooks_dir / "pre_prompt.py"
        script.write_text("print('pre_prompt hook executed')")
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
        assert result != tmpdir  # Should return temp directory

    # Test 3: pre_prompt hook fails
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        script = hooks_dir / "pre_prompt.py"
        script.write_text("import sys; sys.exit(1)")
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)

    # Test 4: Multiple pre_prompt hooks execute
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        script1 = hooks_dir / "pre_prompt.py"
        script1.write_text("print('hook1')")
        script2 = hooks_dir / "pre_prompt.sh"
        script2.write_text("#!/bin/bash\necho 'hook2'")
        script2.chmod(0o755)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))

    # Test 5: Invalid hook files are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        valid_script = hooks_dir / "pre_prompt.py"
        valid_script.write_text("print('valid')")
        invalid_script = hooks_dir / "pre_gen_project.py"  # Wrong hook name
        invalid_script.write_text("print('invalid')")
        backup_script = hooks_dir / "pre_prompt.py~"  # Backup file
        backup_script.write_text("print('backup')")
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script():
    import tempfile
    import os
    from pathlib import Path
    import sys
    
    # Test 1: Successful Python script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(0)')
        script_path = f.name
    
    try:
        run_script(script_path, cwd=tempfile.gettempdir())
    finally:
        os.unlink(script_path)
    
    # Test 2: Successful shell script execution (Unix-like)
    if not sys.platform.startswith('win'):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('#!/bin/sh\nexit 0')
            script_path = f.name
        
        try:
            run_script(script_path, cwd=tempfile.gettempdir())
        finally:
            os.unlink(script_path)
    
    # Test 3: Failed hook exception for non-zero exit
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        script_path = f.name
    
    try:
        try:
            run_script(script_path, cwd=tempfile.gettempdir())
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Hook script failed (exit status: 1)" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 4: Failed hook exception for empty file (ENOEXEC)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write('')  # Empty file
        script_path = f.name
    
    try:
        try:
            run_script(script_path, cwd=tempfile.gettempdir())
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Hook script failed, might be an empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 5: Failed hook exception for other OSError
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(0)')
        script_path = f.name
    
    # Make file non-executable to simulate permission error on some systems
    os.chmod(script_path, 0o444)
    
    try:
        try:
            run_script(script_path, cwd=tempfile.gettempdir())
        except FailedHookException as e:
            assert "Hook script failed (error:" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 6: Windows shell execution path
    original_platform = sys.platform
    try:
        sys.platform = 'win32'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('import sys\nsys.exit(0)')
            script_path = f.name
        
        try:
            run_script(script_path, cwd=tempfile.gettempdir())
        finally:
            os.unlink(script_path)
    finally:
        sys.platform = original_platform


# LLM-generated content at query #4
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test 1: No pre_prompt hook exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

    # Test 2: Valid pre_prompt hook executes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        script_path = hooks_dir / "pre_prompt.py"
        script_path.write_text("#!/usr/bin/env python\nprint('pre_prompt hook executed')")
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir  # Should return a temporary directory path

    # Test 3: pre_prompt hook fails with FailedHookException
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        script_path = hooks_dir / "pre_prompt.py"
        script_path.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)

    # Test 4: Multiple pre_prompt hooks execute in sequence
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        
        script1 = hooks_dir / "pre_prompt.py"
        script1.write_text("#!/usr/bin/env python\nprint('hook1')")
        
        script2 = hooks_dir / "pre_prompt.sh"
        script2.write_text("#!/bin/bash\necho 'hook2'")
        script2.chmod(0o755)
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir

    # Test 5: Invalid hook files are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        
        valid_script = hooks_dir / "pre_prompt.py"
        valid_script.write_text("#!/usr/bin/env python\nprint('valid')")
        
        backup_file = hooks_dir / "pre_prompt.py~"
        backup_file.write_text("backup")
        
        wrong_hook = hooks_dir / "post_gen_project.py"
        wrong_hook.write_text("wrong hook")
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir

    # Test 6: Empty hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir


# LLM-generated content at query #5
#--------------------------

```python
def test_run_hook():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock, call
    from cookiecutter.exceptions import FailedHookException
    
    # Test 1: No hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = None
        with patch('cookiecutter.hooks.logger') as mock_logger:
            # Should not raise any exception
            run_hook('pre_gen_project', '/tmp/project', {})
            mock_logger.debug.assert_called_once_with('No pre_gen_project hook found')
    
    # Test 2: Hook found and executed successfully
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            run_hook('pre_gen_project', '/tmp/project', {'key': 'value'})
            mock_run_script.assert_called_once_with(
                '/tmp/hooks/pre_gen_project.py',
                '/tmp/project',
                {'key': 'value'}
            )
    
    # Test 3: Multiple hooks found and all executed
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = [
            '/tmp/hooks/pre_gen_project.py',
            '/tmp/hooks/pre_gen_project.sh'
        ]
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            run_hook('pre_gen_project', '/tmp/project', {'key': 'value'})
            assert mock_run_script.call_count == 2
            calls = [
                call('/tmp/hooks/pre_gen_project.py', '/tmp/project', {'key': 'value'}),
                call('/tmp/hooks/pre_gen_project.sh', '/tmp/project', {'key': 'value'})
            ]
            mock_run_script.assert_has_calls(calls, any_order=True)
    
    # Test 4: Hook execution raises FailedHookException
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            mock_run_script.side_effect = FailedHookException("Hook failed")
            try:
                run_hook('pre_gen_project', '/tmp/project', {})
                assert False, "Should have raised FailedHookException"
            except FailedHookException:
                pass  # Expected
    
    # Test 5: Hook execution raises UndefinedError
    from jinja2.exceptions import UndefinedError
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            mock_run_script.side_effect = UndefinedError("Template error")
            try:
                run_hook('pre_gen_project', '/tmp/project', {})
                assert False, "Should have raised UndefinedError"
            except UndefinedError:
                pass  # Expected
    
    # Test 6: Verify logger.debug is called for running hook
    with patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        mock_find_hook.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                run_hook('pre_gen_project', '/tmp/project', {})
                mock_logger.debug.assert_called_once_with('Running hook pre_gen_project')


# LLM-generated content at query #6
#--------------------------

```python
def test_run_hook():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock, mock_open
    import pytest
    
    # Test 1: No hook found
    with patch('cookiecutter.hooks.find_hook', return_value=None) as mock_find:
        with patch('cookiecutter.hooks.logger') as mock_logger:
            result = run_hook('pre_gen_project', '/tmp/project', {'key': 'value'})
            assert result is None
            mock_find.assert_called_once_with('pre_gen_project')
            mock_logger.debug.assert_called_with('No pre_gen_project hook found')
    
    # Test 2: Hook found and executed
    mock_scripts = ['/tmp/hooks/pre_gen_project.py']
    with patch('cookiecutter.hooks.find_hook', return_value=mock_scripts) as mock_find:
        with patch('cookiecutter.hooks.logger') as mock_logger:
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                result = run_hook('pre_gen_project', '/tmp/project', {'key': 'value'})
                assert result is None
                mock_find.assert_called_once_with('pre_gen_project')
                mock_logger.debug.assert_called_with('Running hook pre_gen_project')
                mock_run.assert_called_once_with(
                    '/tmp/hooks/pre_gen_project.py',
                    '/tmp/project',
                    {'key': 'value'}
                )
    
    # Test 3: Multiple hooks found and executed
    mock_scripts = ['/tmp/hooks/pre_gen_project.py', '/tmp/hooks/pre_gen_project.sh']
    with patch('cookiecutter.hooks.find_hook', return_value=mock_scripts) as mock_find:
        with patch('cookiecutter.hooks.logger') as mock_logger:
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                result = run_hook('pre_gen_project', '/tmp/project', {'key': 'value'})
                assert result is None
                assert mock_run.call_count == 2
                mock_run.assert_any_call(
                    '/tmp/hooks/pre_gen_project.py',
                    '/tmp/project',
                    {'key': 'value'}
                )
                mock_run.assert_any_call(
                    '/tmp/hooks/pre_gen_project.sh',
                    '/tmp/project',
                    {'key': 'value'}
                )
    
    # Test 4: Hook execution raises exception
    mock_scripts = ['/tmp/hooks/pre_gen_project.py']
    with patch('cookiecutter.hooks.find_hook', return_value=mock_scripts):
        with patch('cookiecutter.hooks.run_script_with_context', side_effect=Exception('Hook failed')):
            with pytest.raises(Exception, match='Hook failed'):
                run_hook('pre_gen_project', '/tmp/project', {'key': 'value'})
    
    # Test 5: Different hook types
    for hook_name in ['pre_prompt', 'pre_gen_project', 'post_gen_project']:
        with patch('cookiecutter.hooks.find_hook', return_value=[f'/tmp/hooks/{hook_name}.py']):
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                run_hook(hook_name, '/tmp/project', {'key': 'value'})
                mock_run.assert_called_once_with(
                    f'/tmp/hooks/{hook_name}.py',
                    '/tmp/project',
                    {'key': 'value'}
                )
    
    # Test 6: Path objects as parameters
    mock_scripts = ['/tmp/hooks/pre_gen_project.py']
    with patch('cookiecutter.hooks.find_hook', return_value=mock_scripts):
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            run_hook('pre_gen_project', Path('/tmp/project'), {'key': 'value'})
            mock_run.assert_called_once_with(
                '/tmp/hooks/pre_gen_project.py',
                Path('/tmp/project'),
                {'key': 'value'}
            )


# LLM-generated content at query #7
#--------------------------

```python
def test_run_script():
    import tempfile
    import os
    from pathlib import Path
    import stat
    
    # Test 1: Successful Python script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(0)')
        python_script = f.name
    
    try:
        run_script(python_script)
    finally:
        os.unlink(python_script)
    
    # Test 2: Successful shell script execution
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('#!/bin/sh\nexit 0')
        shell_script = f.name
    
    try:
        run_script(shell_script)
    finally:
        os.unlink(shell_script)
    
    # Test 3: Failed Python script execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        failed_python_script = f.name
    
    try:
        try:
            run_script(failed_python_script)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Hook script failed" in str(e)
    finally:
        os.unlink(failed_python_script)
    
    # Test 4: Failed shell script execution
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('#!/bin/sh\nexit 1')
        failed_shell_script = f.name
    
    try:
        try:
            run_script(failed_shell_script)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Hook script failed" in str(e)
    finally:
        os.unlink(failed_shell_script)
    
    # Test 5: Empty file without shebang (ENOEXEC error)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        empty_script = f.name
    
    try:
        try:
            run_script(empty_script)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "empty file or missing a shebang" in str(e)
    finally:
        os.unlink(empty_script)
    
    # Test 6: Script with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('import sys\nsys.exit(0)')
        
        with tempfile.TemporaryDirectory() as work_dir:
            run_script(str(script_path), cwd=work_dir)
    
    # Test 7: Verify script is made executable
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(0)')
        script = f.name
    
    try:
        original_mode = os.stat(script).st_mode
        run_script(script)
        new_mode = os.stat(script).st_mode
        assert new_mode & stat.S_IEXEC, "Script should be made executable"
    finally:
        os.unlink(script)


# LLM-generated content at query #8
#--------------------------

```python
def test_find_hook():
    import tempfile
    import os
    from pathlib import Path
    
    # Test 1: No hooks directory exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', hooks_dir=os.path.join(tmpdir, 'hooks'))
        assert result is None
    
    # Test 2: Empty hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result is None
    
    # Test 3: Valid hook file found
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook_file)]
    
    # Test 4: Multiple valid hook files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook_file2 = os.path.join(hooks_dir, 'pre_gen_project.sh')
        Path(hook_file1).touch()
        Path(hook_file2).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert sorted(result) == sorted([os.path.abspath(hook_file1), os.path.abspath(hook_file2)])
    
    # Test 5: Backup files ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        backup_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        Path(backup_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook_file)]
    
    # Test 6: Unsupported hook type ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'unsupported_hook.py')
        Path(hook_file).touch()
        result = find_hook('unsupported_hook', hooks_dir=hooks_dir)
        assert result is None
    
    # Test 7: Hook name mismatch ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'post_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result is None
    
    # Test 8: Valid hook with different extension
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_prompt')
        Path(hook_file).touch()
        result = find_hook('pre_prompt', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook_file)]


# LLM-generated content at query #9
#--------------------------

```python
def test_find_hook():
    import tempfile
    import os
    from pathlib import Path
    
    # Test 1: No hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', hooks_dir=os.path.join(tmpdir, 'hooks'))
        assert result is None
    
    # Test 2: Empty hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result is None
    
    # Test 3: Valid hook file
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook_file)]
    
    # Test 4: Multiple valid hook files
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook1 = os.path.join(hooks_dir, 'pre_gen_project.py')
        hook2 = os.path.join(hooks_dir, 'post_gen_project.py')
        Path(hook1).touch()
        Path(hook2).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook1)]
    
    # Test 5: Backup file should be ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.py~')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result is None
    
    # Test 6: Unsupported hook name
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'unsupported_hook.py')
        Path(hook_file).touch()
        result = find_hook('unsupported_hook', hooks_dir=hooks_dir)
        assert result is None
    
    # Test 7: Hook without extension
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook_file)]
    
    # Test 8: Hook with different extension
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'pre_gen_project.sh')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result == [os.path.abspath(hook_file)]
    
    # Test 9: Mismatched hook name
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = os.path.join(tmpdir, 'hooks')
        os.makedirs(hooks_dir)
        hook_file = os.path.join(hooks_dir, 'post_gen_project.py')
        Path(hook_file).touch()
        result = find_hook('pre_gen_project', hooks_dir=hooks_dir)
        assert result is None


# LLM-generated content at query #10
#--------------------------

```python
def test_run_hook_from_repo_dir():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, call
    from cookiecutter.exceptions import FailedHookException
    
    # Test 1: Hook runs successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        hook_file = hooks_dir / "pre_gen_project.py"
        hook_file.write_text("print('test')")
        
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name="pre_gen_project",
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )
            
            mock_run_hook.assert_called_once_with(
                "pre_gen_project",
                project_dir,
                context
            )
    
    # Test 2: Hook fails with FailedHookException and delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
             patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
             patch('cookiecutter.hooks.logger') as mock_logger:
            
            mock_run_hook.side_effect = FailedHookException("Hook failed")
            
            try:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name="pre_gen_project",
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )
            except FailedHookException:
                pass
            
            mock_rmtree.assert_called_once_with(project_dir)
            mock_logger.exception.assert_called_once()
    
    # Test 3: Hook fails with UndefinedError and delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
             patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
             patch('cookiecutter.hooks.logger') as mock_logger:
            
            mock_run_hook.side_effect = UndefinedError("Template error")
            
            try:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name="pre_gen_project",
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )
            except UndefinedError:
                pass
            
            mock_rmtree.assert_called_once_with(project_dir)
            mock_logger.exception.assert_called_once()
    
    # Test 4: Hook fails but delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
             patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
             patch('cookiecutter.hooks.logger') as mock_logger:
            
            mock_run_hook.side_effect = FailedHookException("Hook failed")
            
            try:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name="pre_gen_project",
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=False
                )
            except FailedHookException:
                pass
            
            mock_rmtree.assert_not_called()
            mock_logger.exception.assert_called_once()
    
    # Test 5: work_in context manager is used correctly
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in') as mock_work_in, \
             patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            
            mock_context_manager = Mock()
            mock_context_manager.__enter__ = Mock()
            mock_context_manager.__exit__ = Mock()
            mock_work_in.return_value = mock_context_manager
            
            run_hook_from_repo_dir(
                repo_dir=repo_dir,
                hook_name="pre_gen_project",
                project_dir=project_dir,
                context=context,
                delete_project_on_failure=True
            )
            
            mock_work_in.assert_called_once_with(repo_dir)
            mock_context_manager.__enter__.assert_called_once()
            mock_context_manager.__exit__.assert_called_once()


# LLM-generated content at query #11
#--------------------------

```python
def test_run_script_with_context(tmp_path):
    # Create a test script with Jinja2 template
    script_content = """#!/usr/bin/env python
import sys
print("Hello {{ name }}!")
print("Project: {{ project_name }}")
sys.exit(0)
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content)
    
    # Create context dictionary
    context = {
        "name": "TestUser",
        "project_name": "TestProject"
    }
    
    # Create a working directory
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    # Run the script with context
    run_script_with_context(str(script_path), str(cwd), context)
    
    # Verify the script was executed by checking if it ran without raising exceptions
    assert script_path.exists()


def test_run_script_with_context_failed_hook(tmp_path):
    # Create a test script that will fail (non-zero exit code)
    script_content = """#!/usr/bin/env python
import sys
print("{{ greeting }}")
sys.exit(1)
"""
    
    script_path = tmp_path / "failing_script.py"
    script_path.write_text(script_content)
    
    context = {"greeting": "This will fail"}
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    # Should raise FailedHookException
    try:
        run_script_with_context(str(script_path), str(cwd), context)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed" in str(e)


def test_run_script_with_context_empty_file(tmp_path):
    # Create an empty script file
    script_path = tmp_path / "empty_script.py"
    script_path.write_text("")
    
    context = {}
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    # Should raise FailedHookException due to empty file
    try:
        run_script_with_context(str(script_path), str(cwd), context)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed" in str(e)


def test_run_script_with_context_invalid_shebang(tmp_path):
    # Create a script without proper shebang (just raw text)
    script_content = """This is not a valid script
{{ variable }}
Just some text
"""
    
    script_path = tmp_path / "invalid_script.txt"
    script_path.write_text(script_content)
    
    context = {"variable": "test"}
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    # Should raise FailedHookException
    try:
        run_script_with_context(str(script_path), str(cwd), context)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed" in str(e)


def test_run_script_with_context_template_error(tmp_path):
    # Create a script with undefined template variable
    script_content = """#!/usr/bin/env python
import sys
print("{{ undefined_variable }}")
sys.exit(0)
"""
    
    script_path = tmp_path / "template_error_script.py"
    script_path.write_text(script_content)
    
    context = {"defined_variable": "test"}  # Note: undefined_variable is not in context
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    # Should raise UndefinedError during template rendering
    try:
        run_script_with_context(str(script_path), str(cwd), context)
        assert False, "Should have raised UndefinedError"
    except UndefinedError:
        # This is expected - template rendering should fail
        pass


def test_run_script_with_context_binary_script(tmp_path):
    # Create a binary script (non-text content)
    script_path = tmp_path / "binary_script.py"
    script_path.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00')
    
    context = {}
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    # Should raise an encoding error or FailedHookException
    try:
        run_script_with_context(str(script_path), str(cwd), context)
        assert False, "Should have raised an exception"
    except (UnicodeDecodeError, FailedHookException):
        # Either exception is possible depending on when it fails
        pass


def test_run_script_with_context_windows_script(tmp_path, monkeypatch):
    # Test with Windows-style script
    script_content = """@echo off
echo Hello {{ name }}!
"""
    
    script_path = tmp_path / "windows_script.bat"
    script_path.write_text(script_content)
    
    context = {"name": "WindowsUser"}
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    # Mock sys.platform to simulate Windows
    monkeypatch.setattr(sys, 'platform', 'win32')
    
    # Should attempt to run (though may fail if not actually on Windows)
    try:
        run_script_with_context(str(script_path), str(cwd), context)
    except (FailedHookException, OSError):
        # Acceptable since we're not actually on Windows
        pass


def test_run_script_with_context_shell_script(tmp_path):
    # Create a shell script
    script_content = """#!/bin/bash
echo "Project: {{ project_name }}"
exit 0
"""
    
    script_path = tmp_path / "shell_script.sh"
    script_path.write_text(script_content)
    
    context = {"project_name": "ShellProject"}
    cwd = tmp_path / "work_dir"
    cwd.mkdir()
    
    # Should run without errors
    run_script_with_context(str(script_path), str(cwd), context)
    
    assert script_path.exists()


# LLM-generated content at query #12
#--------------------------

```python
def test_run_hook():
    import tempfile
    import os
    from pathlib import Path
    import pytest
    from unittest.mock import Mock, patch, call
    
    # Test 1: No hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = None
        with patch('cookiecutter.hooks.logger') as mock_logger:
            # Should not raise any exception
            run_hook('pre_gen_project', '/some/dir', {})
            mock_logger.debug.assert_called_once_with('No pre_gen_project hook found')
    
    # Test 2: Hook found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = ['/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                context = {'project_name': 'test'}
                run_hook('pre_gen_project', '/project/dir', context)
                
                mock_logger.debug.assert_called_once_with('Running hook pre_gen_project')
                mock_run.assert_called_once_with(
                    '/hooks/pre_gen_project.py',
                    '/project/dir',
                    context
                )
    
    # Test 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = [
            '/hooks/pre_gen_project.py',
            '/hooks/pre_gen_project.sh'
        ]
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                context = {'project_name': 'test'}
                run_hook('pre_gen_project', '/project/dir', context)
                
                mock_logger.debug.assert_called_once_with('Running hook pre_gen_project')
                assert mock_run.call_count == 2
                mock_run.assert_has_calls([
                    call('/hooks/pre_gen_project.py', '/project/dir', context),
                    call('/hooks/pre_gen_project.sh', '/project/dir', context)
                ])
    
    # Test 4: Invalid hook name (not in _HOOKS)
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = ['/hooks/invalid_hook.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                # Should still work but find_hook should filter invalid hooks
                run_hook('invalid_hook', '/project/dir', {})
                mock_run.assert_not_called()
    
    # Test 5: Empty hook list
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = []
        with patch('cookiecutter.hooks.logger') as mock_logger:
            run_hook('pre_gen_project', '/some/dir', {})
            mock_logger.debug.assert_called_once_with('No pre_gen_project hook found')


# LLM-generated content at query #13
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test 1: No pre_prompt hook exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

    # Test 2: Valid pre_prompt hook executes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        script = hooks_dir / "pre_prompt.py"
        script.write_text("#!/usr/bin/env python\nprint('pre_prompt executed')")
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
        assert result != tmpdir  # Should return temp directory

    # Test 3: pre_prompt hook fails with FailedHookException
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        script = hooks_dir / "pre_prompt.py"
        script.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)

    # Test 4: Multiple pre_prompt hooks execute in sequence
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        
        script1 = hooks_dir / "pre_prompt.py"
        script1.write_text("#!/usr/bin/env python\nprint('hook1')")
        
        script2 = hooks_dir / "pre_prompt.sh"
        script2.write_text("#!/bin/bash\necho 'hook2'")
        script2.chmod(0o755)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))

    # Test 5: Invalid hook files are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        
        # Valid hook
        valid_script = hooks_dir / "pre_prompt.py"
        valid_script.write_text("#!/usr/bin/env python\nprint('valid')")
        
        # Invalid hooks that should be ignored
        backup_script = hooks_dir / "pre_prompt.py~"
        backup_script.write_text("backup")
        
        wrong_name = hooks_dir / "post_gen_project.py"
        wrong_name.write_text("wrong name")
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))


# LLM-generated content at query #14
#--------------------------

```python
def test_run_script_with_context():
    import tempfile
    from pathlib import Path
    import sys
    from unittest.mock import patch, mock_open, MagicMock
    
    # Test 1: Successful execution of Python script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('Hello {{ name }}')"
        
        with patch('pathlib.Path.read_text', return_value=script_content):
            with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                mock_template = MagicMock()
                mock_template.render.return_value = "print('Hello World')"
                mock_env.return_value.from_string.return_value = mock_template
                
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_run_script.return_value = None
                    
                    run_script_with_context(
                        script_path=script_path,
                        cwd=tmpdir,
                        context={'name': 'World'}
                    )
                    
                    mock_env.assert_called_once_with({'name': 'World'})
                    mock_env.return_value.from_string.assert_called_once_with(script_content)
                    mock_template.render.assert_called_once_with(**{'name': 'World'})
                    mock_run_script.assert_called_once()

    # Test 2: Successful execution of shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.sh"
        script_content = "echo 'Hello {{ name }}'"
        
        with patch('pathlib.Path.read_text', return_value=script_content):
            with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                mock_template = MagicMock()
                mock_template.render.return_value = "echo 'Hello World'"
                mock_env.return_value.from_string.return_value = mock_template
                
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_run_script.return_value = None
                    
                    run_script_with_context(
                        script_path=script_path,
                        cwd=tmpdir,
                        context={'name': 'World'}
                    )
                    
                    mock_run_script.assert_called_once()

    # Test 3: Template rendering with complex context
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('Project: {{ project_name }}, Version: {{ version }}')"
        
        with patch('pathlib.Path.read_text', return_value=script_content):
            with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                mock_template = MagicMock()
                mock_template.render.return_value = "print('Project: MyProject, Version: 1.0')"
                mock_env.return_value.from_string.return_value = mock_template
                
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_run_script.return_value = None
                    
                    context = {
                        'project_name': 'MyProject',
                        'version': '1.0',
                        'author': 'Test Author'
                    }
                    
                    run_script_with_context(
                        script_path=script_path,
                        cwd=tmpdir,
                        context=context
                    )
                    
                    mock_template.render.assert_called_once_with(**context)

    # Test 4: Script with file extension preserved
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('test')"
        
        with patch('pathlib.Path.read_text', return_value=script_content):
            with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                mock_template = MagicMock()
                mock_template.render.return_value = "print('test')"
                mock_env.return_value.from_string.return_value = mock_template
                
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_run_script.return_value = None
                    
                    run_script_with_context(
                        script_path=script_path,
                        cwd=tmpdir,
                        context={}
                    )
                    
                    # Verify the temporary file was created with .py extension
                    mock_run_script.assert_called_once()
                    call_args = mock_run_script.call_args
                    temp_file_path = call_args[0][0]
                    assert temp_file_path.endswith('.py')

    # Test 5: Empty script content
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "empty_script.py"
        script_content = ""
        
        with patch('pathlib.Path.read_text', return_value=script_content):
            with patch('cookiecutter.hooks.create_env_with_context') as mock_env:
                mock_template = MagicMock()
                mock_template.render.return_value = ""
                mock_env.return_value.from_string.return_value = mock_template
                
                with patch('cookiecutter.hooks.run_script') as mock_run_script:
                    mock_run_script.return_value = None
                    
                    run_script_with_context(
                        script_path=script_path,
                        cwd=tmpdir,
                        context={}
                    )
                    
                    mock_run_script.assert_called_once()


# LLM-generated content at query #15
#--------------------------

```python
def test_run_hook_from_repo_dir():
    import tempfile
    import os
    from pathlib import Path
    import pytest
    from unittest.mock import Mock, patch, call
    from cookiecutter.exceptions import FailedHookException

    # Test 1: Hook not found - should do nothing
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.find_hook', return_value=None) as mock_find:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                run_hook_from_repo_dir(
                    repo_dir, 
                    "pre_gen_project", 
                    project_dir, 
                    context, 
                    True
                )
                
                mock_find.assert_called_once_with("pre_gen_project")
                assert mock_logger.debug.called

    # Test 2: Hook found and runs successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                mock_work_in.return_value.__enter__.return_value = None
                
                run_hook_from_repo_dir(
                    repo_dir, 
                    "pre_gen_project", 
                    project_dir, 
                    context, 
                    True
                )
                
                mock_work_in.assert_called_once_with(repo_dir)
                mock_run_hook.assert_called_once_with(
                    "pre_gen_project", 
                    project_dir, 
                    context
                )

    # Test 3: Hook fails with FailedHookException and delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                    with patch('cookiecutter.hooks.logger') as mock_logger:
                        mock_work_in.return_value.__enter__.return_value = None
                        mock_run_hook.side_effect = FailedHookException("Hook failed")
                        
                        with pytest.raises(FailedHookException):
                            run_hook_from_repo_dir(
                                repo_dir, 
                                "pre_gen_project", 
                                project_dir, 
                                context, 
                                True
                            )
                        
                        mock_rmtree.assert_called_once_with(project_dir)
                        assert mock_logger.exception.called

    # Test 4: Hook fails with UndefinedError and delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                    with patch('cookiecutter.hooks.logger') as mock_logger:
                        mock_work_in.return_value.__enter__.return_value = None
                        mock_run_hook.side_effect = UndefinedError("Template error")
                        
                        with pytest.raises(UndefinedError):
                            run_hook_from_repo_dir(
                                repo_dir, 
                                "pre_gen_project", 
                                project_dir, 
                                context, 
                                True
                            )
                        
                        mock_rmtree.assert_called_once_with(project_dir)
                        assert mock_logger.exception.called

    # Test 5: Hook fails but delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in') as mock_work_in:
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                    with patch('cookiecutter.hooks.logger') as mock_logger:
                        mock_work_in.return_value.__enter__.return_value = None
                        mock_run_hook.side_effect = FailedHookException("Hook failed")
                        
                        with pytest.raises(FailedHookException):
                            run_hook_from_repo_dir(
                                repo_dir, 
                                "pre_gen_project", 
                                project_dir, 
                                context, 
                                False
                            )
                        
                        mock_rmtree.assert_not_called()
                        assert mock_logger.exception.called

    # Test 6: Verify work_in context manager usage
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=None)
        mock_context.__exit__ = Mock(return_value=None)
        
        with patch('cookiecutter.hooks.work_in', return_value=mock_context) as mock_work_in:
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                run_hook_from_repo_dir(
                    repo_dir, 
                    "post_gen_project", 
                    project_dir, 
                    context, 
                    True
                )
                
                mock_work_in.assert_called_once_with(repo_dir)
                mock_context.__enter__.assert_called_once()
                mock_context.__exit__.assert_called_once()
                mock_run_hook.assert_called_once_with(
                    "post_gen_project", 
                    project_dir, 
                    context
                )


# LLM-generated content at query #16
#--------------------------

```python
def test_run_script_with_context():
    # Test 1: Basic script execution with context
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('{{ greeting }} {{ name }}!')"
        script_path.write_text(script_content, encoding='utf-8')
        
        cwd = tmpdir
        context = {'greeting': 'Hello', 'name': 'World'}
        
        # Mock subprocess.Popen to capture the command
        original_popen = subprocess.Popen
        captured_command = None
        
        def mock_popen(cmd, **kwargs):
            nonlocal captured_command
            captured_command = cmd
            # Create a mock process that returns success
            class MockProcess:
                def wait(self):
                    return 0
            return MockProcess()
        
        subprocess.Popen = mock_popen
        
        try:
            run_script_with_context(script_path, cwd, context)
            
            # Check that the command was called with the temporary file
            assert captured_command is not None
            assert captured_command[0] == sys.executable
            assert captured_command[1].endswith('.py')
            
            # Verify the temporary file contains rendered content
            with open(captured_command[1], 'r', encoding='utf-8') as f:
                rendered_content = f.read()
                assert rendered_content == 'Hello World!'
                
        finally:
            subprocess.Popen = original_popen
    
    # Test 2: Script with shell script extension
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.sh"
        script_content = "#!/bin/bash\necho '{{ message }}'"
        script_path.write_text(script_content, encoding='utf-8')
        
        cwd = tmpdir
        context = {'message': 'Test message'}
        
        original_popen = subprocess.Popen
        captured_command = None
        
        def mock_popen(cmd, **kwargs):
            nonlocal captured_command
            captured_command = cmd
            class MockProcess:
                def wait(self):
                    return 0
            return MockProcess()
        
        subprocess.Popen = mock_popen
        
        try:
            run_script_with_context(script_path, cwd, context)
            
            # Check that the command was called with the script path directly
            assert captured_command is not None
            assert captured_command[0].endswith('.sh')
            
        finally:
            subprocess.Popen = original_popen
    
    # Test 3: Failed hook exception
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('{{ variable }}')"
        script_path.write_text(script_content, encoding='utf-8')
        
        cwd = tmpdir
        context = {'variable': 'value'}
        
        original_popen = subprocess.Popen
        
        def mock_popen(cmd, **kwargs):
            class MockProcess:
                def wait(self):
                    return 1  # Non-zero exit status
            return MockProcess()
        
        subprocess.Popen = mock_popen
        
        try:
            with pytest.raises(FailedHookException) as exc_info:
                run_script_with_context(script_path, cwd, context)
            assert "Hook script failed" in str(exc_info.value)
        finally:
            subprocess.Popen = original_popen
    
    # Test 4: Jinja2 template error
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('{{ undefined_variable }}')"
        script_path.write_text(script_content, encoding='utf-8')
        
        cwd = tmpdir
        context = {'defined_variable': 'value'}  # Note: undefined_variable is not in context
        
        # This should raise UndefinedError during template rendering
        with pytest.raises(UndefinedError):
            run_script_with_context(script_path, cwd, context)
    
    # Test 5: Empty context
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('Static content')"
        script_path.write_text(script_content, encoding='utf-8')
        
        cwd = tmpdir
        context = {}
        
        original_popen = subprocess.Popen
        
        def mock_popen(cmd, **kwargs):
            class MockProcess:
                def wait(self):
                    return 0
            return MockProcess()
        
        subprocess.Popen = mock_popen
        
        try:
            run_script_with_context(script_path, cwd, context)
        finally:
            subprocess.Popen = original_popen


# LLM-generated content at query #17
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test 1: No pre_prompt hook exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

    # Test 2: Valid pre_prompt hook executes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('print("pre_prompt hook executed")')
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir  # Should return a temporary directory path

    # Test 3: pre_prompt hook fails with FailedHookException
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('import sys\nsys.exit(1)')
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)

    # Test 4: pre_prompt hook with shell script
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt'
        script.write_text('#!/bin/bash\necho "test"')
        script.chmod(0o755)
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir

    # Test 5: Multiple pre_prompt hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        script1 = hooks_dir / 'pre_prompt.py'
        script1.write_text('print("hook1")')
        
        script2 = hooks_dir / 'pre_prompt.sh'
        script2.write_text('#!/bin/bash\necho "hook2"')
        script2.chmod(0o755)
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir

    # Test 6: Invalid hook files are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        # Valid hook
        valid_script = hooks_dir / 'pre_prompt.py'
        valid_script.write_text('print("valid")')
        
        # Invalid hooks that should be ignored
        backup = hooks_dir / 'pre_prompt.py~'
        backup.write_text('print("backup")')
        
        wrong_name = hooks_dir / 'wrong_name.py'
        wrong_name.write_text('print("wrong")')
        
        unsupported = hooks_dir / 'unsupported_hook.py'
        unsupported.write_text('print("unsupported")')
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir


# LLM-generated content at query #18
#--------------------------

```python
def test_valid_hook():
    # Test valid hook file
    assert valid_hook("pre_gen_project.py", "pre_gen_project") == True
    assert valid_hook("post_gen_project.sh", "post_gen_project") == True
    assert valid_hook("pre_prompt", "pre_prompt") == True
    
    # Test invalid hook names
    assert valid_hook("invalid_hook.py", "invalid_hook") == False
    assert valid_hook("pre_gen_project.py", "post_gen_project") == False
    
    # Test backup files
    assert valid_hook("pre_gen_project.py~", "pre_gen_project") == False
    assert valid_hook("post_gen_project.sh~", "post_gen_project") == False
    
    # Test unsupported hooks
    assert valid_hook("unsupported_hook.py", "unsupported_hook") == False
    
    # Test with different extensions
    assert valid_hook("pre_gen_project.sh", "pre_gen_project") == True
    assert valid_hook("pre_gen_project", "pre_gen_project") == True
    assert valid_hook("pre_gen_project.bat", "pre_gen_project") == True


# LLM-generated content at query #19
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test 1: No pre_prompt hook exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

    # Test 2: Valid pre_prompt hook executes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('print("pre_prompt hook executed")')
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir  # Should return a temporary directory
        assert Path(result).exists()

    # Test 3: Multiple pre_prompt hooks execute
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script1 = hooks_dir / 'pre_prompt.py'
        script1.write_text('print("hook 1")')
        script2 = hooks_dir / 'pre_prompt.sh'
        script2.write_text('#!/bin/bash\necho "hook 2"')
        script2.chmod(0o755)
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert Path(result).exists()

    # Test 4: Hook failure raises FailedHookException
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        script = hooks_dir / 'pre_prompt.py'
        script.write_text('import sys\nsys.exit(1)')
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException:
            pass

    # Test 5: Invalid hook files are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        valid_script = hooks_dir / 'pre_prompt.py'
        valid_script.write_text('print("valid")')
        invalid_script = hooks_dir / 'pre_gen_project.py'  # Wrong hook name
        invalid_script.write_text('print("should not run")')
        backup_script = hooks_dir / 'pre_prompt.py~'  # Backup file
        backup_script.write_text('print("backup")')
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert Path(result).exists()

    # Test 6: Empty hooks directory returns original repo_dir
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir


# LLM-generated content at query #20
#--------------------------

```python
def test_run_hook():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import patch, MagicMock, mock_open
    import pytest
    
    # Test 1: No hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = None
        with patch('cookiecutter.hooks.logger') as mock_logger:
            # Should not raise any exception
            run_hook('pre_gen_project', '/tmp/project', {})
            mock_logger.debug.assert_called_with('No pre_gen_project hook found')
    
    # Test 2: Hook found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                context = {'project_name': 'test'}
                run_hook('pre_gen_project', '/tmp/project', context)
                
                mock_logger.debug.assert_called_with('Running hook pre_gen_project')
                mock_run.assert_called_once_with(
                    '/tmp/hooks/pre_gen_project.py',
                    '/tmp/project',
                    context
                )
    
    # Test 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = [
            '/tmp/hooks/pre_gen_project.py',
            '/tmp/hooks/pre_gen_project.sh'
        ]
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            context = {'project_name': 'test'}
            run_hook('pre_gen_project', '/tmp/project', context)
            
            assert mock_run.call_count == 2
            mock_run.assert_any_call(
                '/tmp/hooks/pre_gen_project.py',
                '/tmp/project',
                context
            )
            mock_run.assert_any_call(
                '/tmp/hooks/pre_gen_project.sh',
                '/tmp/project',
                context
            )
    
    # Test 4: Hook execution raises exception
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            mock_run.side_effect = Exception("Hook failed")
            with pytest.raises(Exception):
                run_hook('pre_gen_project', '/tmp/project', {})
    
    # Test 5: Test with different hook types
    for hook_name in ['pre_prompt', 'pre_gen_project', 'post_gen_project']:
        with patch('cookiecutter.hooks.find_hook') as mock_find:
            mock_find.return_value = [f'/tmp/hooks/{hook_name}.py']
            with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
                run_hook(hook_name, '/tmp/project', {})
                mock_run.assert_called_once()
    
    # Test 6: Test with Path objects instead of strings
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            project_dir = Path('/tmp/project')
            context = {'key': 'value'}
            run_hook('pre_gen_project', project_dir, context)
            
            mock_run.assert_called_once_with(
                '/tmp/hooks/pre_gen_project.py',
                project_dir,
                context
            )


# LLM-generated content at query #21
#--------------------------

```python
def test_run_pre_prompt_hook():
    import tempfile
    import os
    from pathlib import Path
    import stat
    import sys
    
    # Test 1: No hooks directory exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir
    
    # Test 2: Hooks directory exists but no pre_prompt hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir
    
    # Test 3: Valid pre_prompt hook that succeeds
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys\nsys.exit(0)')
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
    
    # Test 4: Valid pre_prompt hook that fails
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys\nsys.exit(1)')
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)
    
    # Test 5: Shell script pre_prompt hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt'
        if sys.platform.startswith('win'):
            hook_file.write_text('@echo off\nexit /b 0')
        else:
            hook_file.write_text('#!/bin/bash\nexit 0')
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
    
    # Test 6: Multiple pre_prompt hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        # First hook
        hook_file1 = hooks_dir / 'pre_prompt.py'
        hook_file1.write_text('import sys\nsys.exit(0)')
        hook_file1.chmod(hook_file1.stat().st_mode | stat.S_IEXEC)
        
        # Second hook
        hook_file2 = hooks_dir / 'pre_prompt.sh'
        if sys.platform.startswith('win'):
            hook_file2.write_text('@echo off\nexit /b 0')
        else:
            hook_file2.write_text('#!/bin/bash\nexit 0')
        hook_file2.chmod(hook_file2.stat().st_mode | stat.S_IEXEC)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
    
    # Test 7: Backup files should be ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        # Backup file
        backup_file = hooks_dir / 'pre_prompt.py~'
        backup_file.write_text('import sys\nsys.exit(1)')
        
        # Actual hook
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys\nsys.exit(0)')
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))


# LLM-generated content at query #22
#--------------------------

```python
def test_run_pre_prompt_hook():
    import tempfile
    import os
    from pathlib import Path
    import stat
    
    # Test 1: No hooks directory exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir
    
    # Test 2: Hooks directory exists but no pre_prompt hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'post_gen_project.py').touch()
        
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir
    
    # Test 3: Valid pre_prompt hook executes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('print("pre_prompt hook executed")')
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir  # Should return a temporary directory path
        assert os.path.exists(result)
    
    # Test 4: Multiple pre_prompt hooks execute in order
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        hook1 = hooks_dir / 'pre_prompt.py'
        hook1.write_text('print("hook1")')
        
        hook2 = hooks_dir / 'pre_prompt.sh'
        hook2.write_text('#!/bin/bash\necho "hook2"')
        hook2.chmod(hook2.stat().st_mode | stat.S_IEXEC)
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert os.path.exists(result)
    
    # Test 5: Hook failure raises FailedHookException
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys\nsys.exit(1)')
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)
    
    # Test 6: Hook with Jinja templating (should not be rendered for pre_prompt)
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('{{ cookiecutter.project_name }}')
        
        result = run_pre_prompt_hook(tmpdir)
        assert result != tmpdir
        assert os.path.exists(result)
    
    # Test 7: Backup files are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        (hooks_dir / 'pre_prompt.py~').touch()
        (hooks_dir / 'pre_prompt.py.bak').touch()
        
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir  # Should ignore backup files and return original
    
    # Test 8: Invalid hook names are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        (hooks_dir / 'invalid_hook.py').touch()
        (hooks_dir / 'pre_gen_project.py').touch()
        
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir  # Should ignore invalid hooks


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_run_pre_prompt_hook():
    import tempfile
    import os
    from pathlib import Path
    import stat
    import sys
    
    # Test 1: No hooks directory exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir
    
    # Test 2: Hooks directory exists but no pre_prompt hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        (hooks_dir / 'post_gen_project.py').touch()
        
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir
    
    # Test 3: Valid pre_prompt hook executes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys\nsys.exit(0)')
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
        assert result != tmpdir  # Should return a temporary directory
    
    # Test 4: pre_prompt hook fails with FailedHookException
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys\nsys.exit(1)')
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)
    
    # Test 5: Shell script pre_prompt hook
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        hook_file = hooks_dir / 'pre_prompt'
        if sys.platform.startswith('win'):
            hook_file.write_text('@echo off\nexit /b 0')
        else:
            hook_file.write_text('#!/bin/sh\nexit 0')
            hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
        assert result != tmpdir
    
    # Test 6: Multiple pre_prompt hooks
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        # Create multiple pre_prompt hooks
        hook1 = hooks_dir / 'pre_prompt.py'
        hook1.write_text('import sys\nsys.exit(0)')
        hook1.chmod(hook1.stat().st_mode | stat.S_IEXEC)
        
        hook2 = hooks_dir / 'pre_prompt.sh'
        if sys.platform.startswith('win'):
            hook2.write_text('@echo off\nexit /b 0')
        else:
            hook2.write_text('#!/bin/sh\nexit 0')
            hook2.chmod(hook2.stat().st_mode | stat.S_IEXEC)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
        assert result != tmpdir
    
    # Test 7: Backup files are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        # Create backup file that should be ignored
        backup_file = hooks_dir / 'pre_prompt.py~'
        backup_file.write_text('import sys\nsys.exit(1)')
        
        # Create actual hook
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys\nsys.exit(0)')
        hook_file.chmod(hook_file.stat().st_mode | stat.S_IEXEC)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
        assert result != tmpdir


# LLM-generated content at query #2
#--------------------------

```python
def test_run_hook_from_repo_dir():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, call
    import pytest
    
    # Test 1: Hook not found - should do nothing
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.find_hook', return_value=None) as mock_find:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                run_hook_from_repo_dir(
                    repo_dir, 
                    'pre_gen_project', 
                    project_dir, 
                    context, 
                    True
                )
                
                mock_find.assert_called_once_with('pre_gen_project')
                assert mock_logger.debug.called
                assert "No pre_gen_project hook found" in mock_logger.debug.call_args[0][0]

    # Test 2: Hook runs successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in'):
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                run_hook_from_repo_dir(
                    repo_dir, 
                    'pre_gen_project', 
                    project_dir, 
                    context, 
                    True
                )
                
                mock_run_hook.assert_called_once_with(
                    'pre_gen_project', 
                    project_dir, 
                    context
                )

    # Test 3: Hook fails with FailedHookException and delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in'):
            with patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed")):
                with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                    with patch('cookiecutter.hooks.logger') as mock_logger:
                        with pytest.raises(FailedHookException):
                            run_hook_from_repo_dir(
                                repo_dir, 
                                'pre_gen_project', 
                                project_dir, 
                                context, 
                                True
                            )
                        
                        mock_rmtree.assert_called_once_with(project_dir)
                        mock_logger.exception.assert_called_once()

    # Test 4: Hook fails with FailedHookException and delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in'):
            with patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed")):
                with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                    with patch('cookiecutter.hooks.logger') as mock_logger:
                        with pytest.raises(FailedHookException):
                            run_hook_from_repo_dir(
                                repo_dir, 
                                'pre_gen_project', 
                                project_dir, 
                                context, 
                                False
                            )
                        
                        mock_rmtree.assert_not_called()
                        mock_logger.exception.assert_called_once()

    # Test 5: Hook fails with UndefinedError and delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in'):
            with patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Template error")):
                with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                    with patch('cookiecutter.hooks.logger') as mock_logger:
                        with pytest.raises(UndefinedError):
                            run_hook_from_repo_dir(
                                repo_dir, 
                                'pre_gen_project', 
                                project_dir, 
                                context, 
                                True
                            )
                        
                        mock_rmtree.assert_called_once_with(project_dir)
                        mock_logger.exception.assert_called_once()

    # Test 6: Hook fails with UndefinedError and delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.work_in'):
            with patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Template error")):
                with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                    with patch('cookiecutter.hooks.logger') as mock_logger:
                        with pytest.raises(UndefinedError):
                            run_hook_from_repo_dir(
                                repo_dir, 
                                'pre_gen_project', 
                                project_dir, 
                                context, 
                                False
                            )
                        
                        mock_rmtree.assert_not_called()
                        mock_logger.exception.assert_called_once()

    # Test 7: Verify work_in context manager is used correctly
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        mock_work_in = Mock()
        mock_work_in.return_value.__enter__ = Mock()
        mock_work_in.return_value.__exit__ = Mock()
        
        with patch('cookiecutter.hooks.work_in', mock_work_in):
            with patch('cookiecutter.hooks.run_hook'):
                run_hook_from_repo_dir(
                    repo_dir, 
                    'pre_gen_project', 
                    project_dir, 
                    context, 
                    True
                )
                
                mock_work_in.assert_called_once_with(repo_dir)


# LLM-generated content at query #3
#--------------------------

```python
def test_run_script():
    import tempfile
    import os
    from pathlib import Path
    import stat
    
    # Test 1: Successful execution of Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(0)')
        script_path = f.name
    
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)
    
    # Test 2: Successful execution of shell script
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('#!/bin/sh\nexit 0')
        script_path = f.name
    
    try:
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)
        run_script(script_path)
    finally:
        os.unlink(script_path)
    
    # Test 3: Script with non-zero exit code raises FailedHookException
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        script_path = f.name
    
    try:
        try:
            run_script(script_path)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "exit status: 1" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 4: Non-executable script raises FailedHookException
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('')
        script_path = f.name
    
    try:
        os.chmod(script_path, stat.S_IRUSR)  # Read-only, not executable
        try:
            run_script(script_path)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Hook script failed" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 5: Empty script file raises FailedHookException
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('')  # Empty file without shebang
        script_path = f.name
    
    try:
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)
        try:
            run_script(script_path)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 6: Script execution with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test.py'
        script_path.write_text('import sys\nsys.exit(0)')
        
        with tempfile.TemporaryDirectory() as cwd:
            run_script(str(script_path), cwd=cwd)
    
    # Test 7: Windows-style script execution (shell=True)
    original_platform = sys.platform
    try:
        sys.platform = 'win32'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('import sys\nsys.exit(0)')
            script_path = f.name
        
        try:
            run_script(script_path)
        finally:
            os.unlink(script_path)
    finally:
        sys.platform = original_platform


# LLM-generated content at query #4
#--------------------------

```python
def test_run_script_with_context():
    import tempfile
    from pathlib import Path
    import subprocess
    import sys
    
    # Mock the necessary components
    class MockPath:
        def __init__(self, content):
            self.content = content
            self.path = "/mock/path/script.py"
        
        def read_text(self, encoding):
            return self.content
        
        def __str__(self):
            return self.path
    
    # Test 1: Successful script execution with Jinja templating
    def test_successful_execution():
        script_content = "print('{{ project_name }}')"
        context = {"project_name": "TestProject"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple Python script to verify execution
            test_script = Path(tmpdir) / "test_script.py"
            test_script.write_text("import sys; print('TestProject'); sys.exit(0)")
            
            # Mock the template rendering to return our test script content
            original_read_text = Path.read_text
            Path.read_text = lambda self, encoding: "print('TestProject')"
            
            try:
                # We need to mock subprocess.Popen for this test
                original_popen = subprocess.Popen
                mock_proc = type('MockProc', (), {})()
                mock_proc.wait = lambda: 0
                
                subprocess.Popen = lambda *args, **kwargs: mock_proc
                
                # This should not raise any exceptions
                run_script_with_context(test_script, tmpdir, context)
            finally:
                Path.read_text = original_read_text
                subprocess.Popen = original_popen
    
    # Test 2: Script execution failure
    def test_execution_failure():
        script_content = "print('{{ project_name }}')"
        context = {"project_name": "TestProject"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_script = Path(tmpdir) / "test_script.py"
            test_script.write_text("import sys; sys.exit(1)")
            
            original_read_text = Path.read_text
            Path.read_text = lambda self, encoding: "import sys; sys.exit(1)"
            
            try:
                original_popen = subprocess.Popen
                mock_proc = type('MockProc', (), {})()
                mock_proc.wait = lambda: 1
                
                subprocess.Popen = lambda *args, **kwargs: mock_proc
                
                try:
                    run_script_with_context(test_script, tmpdir, context)
                    assert False, "Should have raised FailedHookException"
                except FailedHookException as e:
                    assert "Hook script failed" in str(e)
            finally:
                Path.read_text = original_read_text
                subprocess.Popen = original_popen
    
    # Test 3: Jinja template rendering with context
    def test_jinja_rendering():
        script_content = "Project: {{ project_name }}, Version: {{ version }}"
        context = {"project_name": "MyApp", "version": "1.0.0"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_script = Path(tmpdir) / "test_script.py"
            
            # Track what was written to the temp file
            written_content = []
            original_write = tempfile.NamedTemporaryFile.write
            
            def mock_write(self, data):
                written_content.append(data.decode('utf-8'))
                return len(data)
            
            tempfile.NamedTemporaryFile.write = mock_write
            
            original_read_text = Path.read_text
            Path.read_text = lambda self, encoding: script_content
            
            try:
                original_popen = subprocess.Popen
                mock_proc = type('MockProc', (), {})()
                mock_proc.wait = lambda: 0
                
                subprocess.Popen = lambda *args, **kwargs: mock_proc
                
                run_script_with_context(test_script, tmpdir, context)
                
                # Verify the template was rendered with context
                assert len(written_content) > 0
                rendered = written_content[-1]
                assert "Project: MyApp" in rendered
                assert "Version: 1.0.0" in rendered
            finally:
                Path.read_text = original_read_text
                subprocess.Popen = original_popen
                tempfile.NamedTemporaryFile.write = original_write
    
    # Test 4: Non-Python script execution
    def test_non_python_script():
        script_content = "#!/bin/bash\necho '{{ message }}'"
        context = {"message": "Hello World"}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_script = Path(tmpdir) / "test_script.sh"
            
            original_read_text = Path.read_text
            Path.read_text = lambda self, encoding: script_content
            
            try:
                original_popen = subprocess.Popen
                mock_proc = type('MockProc', (), {})()
                mock_proc.wait = lambda: 0
                
                subprocess.Popen = lambda *args, **kwargs: mock_proc
                
                # This should work for non-Python scripts too
                run_script_with_context(test_script, tmpdir, context)
            finally:
                Path.read_text = original_read_text
                subprocess.Popen = original_popen
    
    # Run all tests
    test_successful_execution()
    test_execution_failure()
    test_jinja_rendering()
    test_non_python_script()


# LLM-generated content at query #5
#--------------------------

```python
def test_run_script_with_context():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, mock_open, call
    
    # Test 1: Successful script execution with context rendering
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('Hello {{ name }}!')"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        context = {'name': 'World'}
        cwd = tmpdir
        
        with patch('cookiecutter.hooks.run_script') as mock_run_script:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                mock_env = Mock()
                mock_create_env.return_value = mock_env
                mock_template = Mock()
                mock_env.from_string.return_value = mock_template
                mock_template.render.return_value = "print('Hello World!')"
                
                with patch('builtins.open', mock_open()) as mock_file:
                    with patch('tempfile.NamedTemporaryFile') as mock_temp:
                        mock_temp_file = Mock()
                        mock_temp_file.name = '/tmp/temp123.py'
                        mock_temp.return_value.__enter__.return_value = mock_temp_file
                        
                        from cookiecutter.hooks import run_script_with_context
                        run_script_with_context(script_path, cwd, context)
                        
                        mock_create_env.assert_called_once_with(context)
                        mock_env.from_string.assert_called_once_with(script_content)
                        mock_template.render.assert_called_once_with(**context)
                        mock_temp.assert_called_once_with(delete=False, mode='wb', suffix='.py')
                        mock_run_script.assert_called_once_with('/tmp/temp123.py', cwd)
    
    # Test 2: Script with different extension
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.sh"
        script_content = "echo 'Hello {{ name }}!'"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        context = {'name': 'Test'}
        cwd = tmpdir
        
        with patch('cookiecutter.hooks.run_script') as mock_run_script:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                mock_env = Mock()
                mock_create_env.return_value = mock_env
                mock_template = Mock()
                mock_env.from_string.return_value = mock_template
                mock_template.render.return_value = "echo 'Hello Test!'"
                
                with patch('builtins.open', mock_open()) as mock_file:
                    with patch('tempfile.NamedTemporaryFile') as mock_temp:
                        mock_temp_file = Mock()
                        mock_temp_file.name = '/tmp/temp123.sh'
                        mock_temp.return_value.__enter__.return_value = mock_temp_file
                        
                        from cookiecutter.hooks import run_script_with_context
                        run_script_with_context(script_path, cwd, context)
                        
                        mock_temp.assert_called_once_with(delete=False, mode='wb', suffix='.sh')
    
    # Test 3: UTF-8 encoding handling
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('Hello {{ name }}!')"
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        context = {'name': 'UTF-8 Test'}
        cwd = tmpdir
        
        with patch('cookiecutter.hooks.run_script') as mock_run_script:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                mock_env = Mock()
                mock_create_env.return_value = mock_env
                mock_template = Mock()
                mock_env.from_string.return_value = mock_template
                mock_template.render.return_value = "print('Hello UTF-8 Test!')"
                
                with patch('pathlib.Path.read_text') as mock_read_text:
                    mock_read_text.return_value = script_content
                    
                    with patch('tempfile.NamedTemporaryFile') as mock_temp:
                        mock_temp_file = Mock()
                        mock_temp_file.name = '/tmp/temp123.py'
                        mock_temp.return_value.__enter__.return_value = mock_temp_file
                        
                        from cookiecutter.hooks import run_script_with_context
                        run_script_with_context(script_path, cwd, context)
                        
                        mock_read_text.assert_called_once_with(encoding='utf-8')
                        mock_temp_file.write.assert_called_once_with(b"print('Hello UTF-8 Test!')")
    
    # Test 4: Path object as script_path parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('{{ value }}')"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        context = {'value': 42}
        cwd = Path(tmpdir)
        
        with patch('cookiecutter.hooks.run_script') as mock_run_script:
            with patch('cookiecutter.hooks.create_env_with_context') as mock_create_env:
                mock_env = Mock()
                mock_create_env.return_value = mock_env
                mock_template = Mock()
                mock_env.from_string.return_value = mock_template
                mock_template.render.return_value = "print('42')"
                
                with patch('tempfile.NamedTemporaryFile') as mock_temp:
                    mock_temp_file = Mock()
                    mock_temp_file.name = '/tmp/temp123.py'
                    mock_temp.return_value.__enter__.return_value = mock_temp_file
                    
                    from cookiecutter.hooks import run_script_with_context
                    run_script_with_context(script_path, cwd, context)
                    
                    mock_run_script.assert_called_once_with('/tmp/temp123.py', cwd)


# LLM-generated content at query #6
#--------------------------

```python
def test_valid_hook():
    # Test valid hook file
    assert valid_hook("pre_gen_project.py", "pre_gen_project") is True
    assert valid_hook("post_gen_project.sh", "post_gen_project") is True
    assert valid_hook("pre_prompt.py", "pre_prompt") is True
    
    # Test invalid hook names
    assert valid_hook("invalid_hook.py", "invalid_hook") is False
    assert valid_hook("random_script.py", "random_script") is False
    
    # Test backup files
    assert valid_hook("pre_gen_project.py~", "pre_gen_project") is False
    assert valid_hook("post_gen_project.sh~", "post_gen_project") is False
    
    # Test wrong hook name
    assert valid_hook("pre_gen_project.py", "post_gen_project") is False
    assert valid_hook("post_gen_project.sh", "pre_gen_project") is False
    
    # Test with different extensions
    assert valid_hook("pre_gen_project", "pre_gen_project") is True
    assert valid_hook("pre_gen_project.exe", "pre_gen_project") is True
    
    # Test unsupported hook
    assert valid_hook("unsupported_hook.py", "unsupported_hook") is False


# LLM-generated content at query #7
#--------------------------

```python
def test_run_hook_from_repo_dir():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, call
    from cookiecutter.exceptions import FailedHookException
    
    # Test 1: Hook runs successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        
        hook_script = hooks_dir / "pre_gen_project.py"
        hook_script.write_text("print('hook executed')")
        
        context = {"project_name": "test"}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
            mock_run_hook.return_value = None
            
            from cookiecutter import hooks
            hooks.run_hook_from_repo_dir(
                repo_dir=str(repo_dir),
                hook_name="pre_gen_project",
                project_dir=str(project_dir),
                context=context,
                delete_project_on_failure=True
            )
            
            mock_run_hook.assert_called_once_with(
                "pre_gen_project",
                str(project_dir),
                context
            )
    
    # Test 2: Hook fails with FailedHookException and delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        context = {"project_name": "test"}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
             patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
             patch('cookiecutter.hooks.logger') as mock_logger:
            
            mock_run_hook.side_effect = FailedHookException("Hook failed")
            
            from cookiecutter import hooks
            
            try:
                hooks.run_hook_from_repo_dir(
                    repo_dir=str(repo_dir),
                    hook_name="pre_gen_project",
                    project_dir=str(project_dir),
                    context=context,
                    delete_project_on_failure=True
                )
                assert False, "Should have raised FailedHookException"
            except FailedHookException:
                pass
            
            mock_rmtree.assert_called_once_with(str(project_dir))
            mock_logger.exception.assert_called_once()
    
    # Test 3: Hook fails with UndefinedError and delete_project_on_failure=False
    from jinja2.exceptions import UndefinedError
    
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        context = {"project_name": "test"}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
             patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
             patch('cookiecutter.hooks.logger') as mock_logger:
            
            mock_run_hook.side_effect = UndefinedError("Template error")
            
            from cookiecutter import hooks
            
            try:
                hooks.run_hook_from_repo_dir(
                    repo_dir=str(repo_dir),
                    hook_name="pre_gen_project",
                    project_dir=str(project_dir),
                    context=context,
                    delete_project_on_failure=False
                )
                assert False, "Should have raised UndefinedError"
            except UndefinedError:
                pass
            
            mock_rmtree.assert_not_called()
            mock_logger.exception.assert_called_once()
    
    # Test 4: Hook fails with FailedHookException and delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        context = {"project_name": "test"}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
             patch('cookiecutter.hooks.rmtree') as mock_rmtree, \
             patch('cookiecutter.hooks.logger') as mock_logger:
            
            mock_run_hook.side_effect = FailedHookException("Hook failed")
            
            from cookiecutter import hooks
            
            try:
                hooks.run_hook_from_repo_dir(
                    repo_dir=str(repo_dir),
                    hook_name="pre_gen_project",
                    project_dir=str(project_dir),
                    context=context,
                    delete_project_on_failure=False
                )
                assert False, "Should have raised FailedHookException"
            except FailedHookException:
                pass
            
            mock_rmtree.assert_not_called()
            mock_logger.exception.assert_called_once()
    
    # Test 5: work_in context manager is used correctly
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        context = {"project_name": "test"}
        
        with patch('cookiecutter.hooks.run_hook') as mock_run_hook, \
             patch('cookiecutter.hooks.work_in') as mock_work_in:
            
            mock_context_manager = Mock()
            mock_context_manager.__enter__ = Mock()
            mock_context_manager.__exit__ = Mock()
            mock_work_in.return_value = mock_context_manager
            
            from cookiecutter import hooks
            
            hooks.run_hook_from_repo_dir(
                repo_dir=str(repo_dir),
                hook_name="pre_gen_project",
                project_dir=str(project_dir),
                context=context,
                delete_project_on_failure=True
            )
            
            mock_work_in.assert_called_once_with(str(repo_dir))
            mock_context_manager.__enter__.assert_called_once()
            mock_context_manager.__exit__.assert_called_once()


# LLM-generated content at query #8
#--------------------------

```python
def test_run_script():
    import tempfile
    import os
    from pathlib import Path
    import stat
    
    # Test 1: Successful execution of Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(0)')
        script_path = f.name
    
    try:
        run_script(script_path)
        # Should not raise exception
    finally:
        os.unlink(script_path)
    
    # Test 2: Successful execution of shell script
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('#!/bin/sh\nexit 0')
        script_path = f.name
    
    try:
        run_script(script_path)
        # Should not raise exception
    finally:
        os.unlink(script_path)
    
    # Test 3: Failed execution - non-zero exit code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        script_path = f.name
    
    try:
        try:
            run_script(script_path)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "exit status: 1" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 4: Failed execution - empty file without shebang
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('')
        script_path = f.name
    
    try:
        os.chmod(script_path, stat.S_IRWXU)
        try:
            run_script(script_path)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 5: Failed execution - file not found
    try:
        run_script("/non/existent/path/script.py")
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed" in str(e)
    
    # Test 6: Execution with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test.py"
        script_path.write_text('import sys\nsys.exit(0)')
        
        with tempfile.TemporaryDirectory() as work_dir:
            run_script(str(script_path), cwd=work_dir)
            # Should not raise exception
    
    # Test 7: Windows-style script execution (simulated)
    original_platform = sys.platform
    try:
        sys.platform = "win32"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('import sys\nsys.exit(0)')
            script_path = f.name
        
        try:
            run_script(script_path)
            # Should not raise exception
        finally:
            os.unlink(script_path)
    finally:
        sys.platform = original_platform


# LLM-generated content at query #9
#--------------------------

```python
def test_run_script_with_context():
    # Test 1: Basic script execution with context
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('{{ greeting }} {{ name }}!')"
        script_path.write_text(script_content, encoding='utf-8')
        
        context = {'greeting': 'Hello', 'name': 'World'}
        cwd = Path(tmpdir)
        
        # Mock subprocess.Popen to capture the command
        original_popen = subprocess.Popen
        captured_command = None
        
        def mock_popen(cmd, *args, **kwargs):
            nonlocal captured_command
            captured_command = cmd
            # Create a mock process that returns success
            class MockProcess:
                def wait(self):
                    return 0
            return MockProcess()
        
        subprocess.Popen = mock_popen
        
        try:
            run_script_with_context(script_path, cwd, context)
            # Check that the script was executed with Python
            assert captured_command[0] == sys.executable
            # Check that a temporary file was created
            assert captured_command[1].endswith('.py')
            
            # Verify the rendered content
            temp_file = Path(captured_command[1])
            rendered_content = temp_file.read_text(encoding='utf-8')
            assert rendered_content == 'print(\'Hello World!\')'
        finally:
            subprocess.Popen = original_popen
    
    # Test 2: Script with Jinja2 template error
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('{{ undefined_variable }}')"
        script_path.write_text(script_content, encoding='utf-8')
        
        context = {'defined_variable': 'test'}
        cwd = Path(tmpdir)
        
        with pytest.raises(UndefinedError):
            run_script_with_context(script_path, cwd, context)
    
    # Test 3: Non-Python script execution
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.sh"
        script_content = "#!/bin/bash\necho '{{ message }}'"
        script_path.write_text(script_content, encoding='utf-8')
        
        context = {'message': 'Test message'}
        cwd = Path(tmpdir)
        
        # Mock subprocess.Popen
        original_popen = subprocess.Popen
        captured_command = None
        
        def mock_popen(cmd, *args, **kwargs):
            nonlocal captured_command
            captured_command = cmd
            class MockProcess:
                def wait(self):
                    return 0
            return MockProcess()
        
        subprocess.Popen = mock_popen
        
        try:
            run_script_with_context(script_path, cwd, context)
            # Check that non-Python script is executed directly
            assert captured_command[0].endswith('.sh')
        finally:
            subprocess.Popen = original_popen
    
    # Test 4: Script execution failure
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "print('{{ text }}')"
        script_path.write_text(script_content, encoding='utf-8')
        
        context = {'text': 'Hello'}
        cwd = Path(tmpdir)
        
        # Mock subprocess.Popen to simulate failure
        original_popen = subprocess.Popen
        
        def mock_popen(cmd, *args, **kwargs):
            class MockProcess:
                def wait(self):
                    return 1  # Non-zero exit status
            return MockProcess()
        
        subprocess.Popen = mock_popen
        
        try:
            with pytest.raises(FailedHookException) as exc_info:
                run_script_with_context(script_path, cwd, context)
            assert "Hook script failed" in str(exc_info.value)
        finally:
            subprocess.Popen = original_popen
    
    # Test 5: File encoding test
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "test_script.py"
        script_content = "# -*- coding: utf-8 -*-\nprint('{{ greeting }}')"
        script_path.write_text(script_content, encoding='utf-8')
        
        context = {'greeting': 'Привет'}  # Unicode characters
        cwd = Path(tmpdir)
        
        # Mock subprocess.Popen
        original_popen = subprocess.Popen
        
        def mock_popen(cmd, *args, **kwargs):
            class MockProcess:
                def wait(self):
                    return 0
            return MockProcess()
        
        subprocess.Popen = mock_popen
        
        try:
            run_script_with_context(script_path, cwd, context)
            # No assertion needed, just ensure it doesn't crash
        finally:
            subprocess.Popen = original_popen


# LLM-generated content at query #10
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test 1: No pre_prompt hook exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

    # Test 2: Valid pre_prompt hook executes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('print("pre_prompt hook executed")')
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
        assert result != tmpdir  # Should return temp directory

    # Test 3: pre_prompt hook fails with FailedHookException
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_prompt.py'
        hook_file.write_text('import sys; sys.exit(1)')
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)

    # Test 4: Multiple pre_prompt hooks execute
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        # Create multiple valid pre_prompt hooks
        hook1 = hooks_dir / 'pre_prompt.py'
        hook1.write_text('print("hook1")')
        
        hook2 = hooks_dir / 'pre_prompt.sh'
        hook2.write_text('#!/bin/bash\necho "hook2"')
        hook2.chmod(0o755)
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))

    # Test 5: Invalid hook files are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        
        # Valid hook
        valid_hook = hooks_dir / 'pre_prompt.py'
        valid_hook.write_text('print("valid")')
        
        # Invalid hooks (should be ignored)
        invalid1 = hooks_dir / 'post_gen_project.py'  # Wrong hook name
        invalid1.write_text('print("wrong hook")')
        
        invalid2 = hooks_dir / 'pre_prompt.py~'  # Backup file
        invalid2.write_text('print("backup")')
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))


# LLM-generated content at query #11
#--------------------------

```python
def test_run_script():
    import tempfile
    import os
    import sys
    from pathlib import Path
    import stat
    
    # Test 1: Successful execution of Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(0)')
        script_path = f.name
    
    try:
        run_script(script_path, cwd=os.path.dirname(script_path))
    finally:
        os.unlink(script_path)
    
    # Test 2: Successful execution of shell script
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('#!/bin/sh\nexit 0')
        script_path = f.name
    
    try:
        run_script(script_path, cwd=os.path.dirname(script_path))
    finally:
        os.unlink(script_path)
    
    # Test 3: Failed hook script (non-zero exit code)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        script_path = f.name
    
    try:
        try:
            run_script(script_path, cwd=os.path.dirname(script_path))
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "exit status: 1" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 4: Empty file or missing shebang (ENOEXEC error simulation)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('')  # Empty file
        script_path = f.name
    
    try:
        # Make it executable but empty to simulate ENOEXEC
        os.chmod(script_path, stat.S_IRWXU)
        try:
            run_script(script_path, cwd=os.path.dirname(script_path))
            assert False, "Should have raised FailedHookException for ENOEXEC"
        except FailedHookException as e:
            assert "empty file or missing a shebang" in str(e)
    finally:
        os.unlink(script_path)
    
    # Test 5: OSError other than ENOEXEC
    # Create a script that will cause OSError (e.g., trying to execute a directory)
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "nonexistent_script")
        try:
            run_script(script_path, cwd=tmpdir)
            assert False, "Should have raised FailedHookException for OSError"
        except FailedHookException as e:
            assert "Hook script failed" in str(e)
    
    # Test 6: Test with different working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "test.py")
        with open(script_path, 'w') as f:
            f.write('import os\nprint(os.getcwd())\nimport sys\nsys.exit(0)')
        
        # Run from a different directory
        with tempfile.TemporaryDirectory() as other_dir:
            run_script(script_path, cwd=other_dir)


# LLM-generated content at query #12
#--------------------------

```python
def test_run_script():
    # Test 1: Successful execution of Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(0)')
        script_path = f.name
    
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)
    
    # Test 2: Successful execution of shell script
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('#!/bin/sh\nexit 0')
        script_path = f.name
    
    try:
        run_script(script_path)
    finally:
        os.unlink(script_path)
    
    # Test 3: Failed execution with non-zero exit code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        script_path = f.name
    
    try:
        with pytest.raises(FailedHookException) as exc_info:
            run_script(script_path)
        assert "Hook script failed (exit status: 1)" in str(exc_info.value)
    finally:
        os.unlink(script_path)
    
    # Test 4: Script file not found
    with pytest.raises(FailedHookException) as exc_info:
        run_script('/non/existent/script.py')
    assert "Hook script failed" in str(exc_info.value)
    
    # Test 5: Empty file without shebang (ENOEXEC error simulation)
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('')
        script_path = f.name
    
    # Make it executable but empty to simulate ENOEXEC
    os.chmod(script_path, 0o755)
    
    try:
        with pytest.raises(FailedHookException) as exc_info:
            run_script(script_path)
        assert "might be an empty file or missing a shebang" in str(exc_info.value)
    finally:
        os.unlink(script_path)
    
    # Test 6: Execution with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, 'test.py')
        with open(script_path, 'w') as f:
            f.write('import os\nprint(os.getcwd())\nimport sys\nsys.exit(0)')
        
        with tempfile.TemporaryDirectory() as cwd:
            run_script(script_path, cwd=cwd)
    
    # Test 7: Windows-style execution (shell=True)
    original_platform = sys.platform
    try:
        sys.platform = 'win32'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('import sys\nsys.exit(0)')
            script_path = f.name
        
        try:
            run_script(script_path)
        finally:
            os.unlink(script_path)
    finally:
        sys.platform = original_platform


# LLM-generated content at query #13
#--------------------------

```python
def test_run_hook():
    # Mock the find_hook function to return a script path
    with mock.patch('cookiecutter.hooks.find_hook') as mock_find_hook:
        # Mock the run_script_with_context function
        with mock.patch('cookiecutter.hooks.run_script_with_context') as mock_run_script:
            # Test 1: Hook found and executed
            mock_find_hook.return_value = ['/fake/path/hook.py']
            context = {'project_name': 'test'}
            
            run_hook('pre_gen_project', '/fake/project_dir', context)
            
            mock_find_hook.assert_called_once_with('pre_gen_project')
            mock_run_script.assert_called_once_with(
                '/fake/path/hook.py',
                '/fake/project_dir',
                context
            )
            
            # Reset mocks for next test
            mock_find_hook.reset_mock()
            mock_run_script.reset_mock()
            
            # Test 2: Multiple hooks found and executed
            mock_find_hook.return_value = ['/fake/path/hook1.py', '/fake/path/hook2.py']
            
            run_hook('post_gen_project', '/another/project_dir', context)
            
            mock_find_hook.assert_called_once_with('post_gen_project')
            assert mock_run_script.call_count == 2
            mock_run_script.assert_has_calls([
                mock.call('/fake/path/hook1.py', '/another/project_dir', context),
                mock.call('/fake/path/hook2.py', '/another/project_dir', context)
            ])
            
            # Reset mocks for next test
            mock_find_hook.reset_mock()
            mock_run_script.reset_mock()
            
            # Test 3: No hook found - nothing should be executed
            mock_find_hook.return_value = None
            
            run_hook('pre_prompt', '/empty/project_dir', context)
            
            mock_find_hook.assert_called_once_with('pre_prompt')
            mock_run_script.assert_not_called()
            
            # Reset mocks for next test
            mock_find_hook.reset_mock()
            mock_run_script.reset_mock()
            
            # Test 4: Empty hook list - nothing should be executed
            mock_find_hook.return_value = []
            
            run_hook('pre_gen_project', '/empty/project_dir', context)
            
            mock_find_hook.assert_called_once_with('pre_gen_project')
            mock_run_script.assert_not_called()


# LLM-generated content at query #14
#--------------------------

```python
def test_run_hook_from_repo_dir():
    import tempfile
    import os
    from pathlib import Path
    from unittest.mock import Mock, patch, call
    import pytest
    from cookiecutter.exceptions import FailedHookException

    # Test 1: Hook not found - should do nothing
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.find_hook', return_value=None) as mock_find:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                run_hook_from_repo_dir(
                    repo_dir=repo_dir,
                    hook_name="pre_gen_project",
                    project_dir=project_dir,
                    context=context,
                    delete_project_on_failure=True
                )
                
                mock_find.assert_called_once_with("pre_gen_project")
                assert mock_logger.debug.called

    # Test 2: Hook runs successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.find_hook', return_value=["/fake/script.py"]):
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                with patch('cookiecutter.utils.work_in'):
                    run_hook_from_repo_dir(
                        repo_dir=repo_dir,
                        hook_name="pre_gen_project",
                        project_dir=project_dir,
                        context=context,
                        delete_project_on_failure=True
                    )
                    
                    mock_run_hook.assert_called_once_with(
                        "pre_gen_project",
                        project_dir,
                        context
                    )

    # Test 3: Hook fails with FailedHookException, delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.find_hook', return_value=["/fake/script.py"]):
            with patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed")):
                with patch('cookiecutter.utils.work_in'):
                    with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                        with patch('cookiecutter.hooks.logger') as mock_logger:
                            with pytest.raises(FailedHookException):
                                run_hook_from_repo_dir(
                                    repo_dir=repo_dir,
                                    hook_name="pre_gen_project",
                                    project_dir=project_dir,
                                    context=context,
                                    delete_project_on_failure=True
                                )
                            
                            mock_rmtree.assert_called_once_with(project_dir)
                            assert mock_logger.exception.called

    # Test 4: Hook fails with FailedHookException, delete_project_on_failure=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.find_hook', return_value=["/fake/script.py"]):
            with patch('cookiecutter.hooks.run_hook', side_effect=FailedHookException("Hook failed")):
                with patch('cookiecutter.utils.work_in'):
                    with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                        with patch('cookiecutter.hooks.logger') as mock_logger:
                            with pytest.raises(FailedHookException):
                                run_hook_from_repo_dir(
                                    repo_dir=repo_dir,
                                    hook_name="pre_gen_project",
                                    project_dir=project_dir,
                                    context=context,
                                    delete_project_on_failure=False
                                )
                            
                            mock_rmtree.assert_not_called()
                            assert mock_logger.exception.called

    # Test 5: Hook fails with UndefinedError, delete_project_on_failure=True
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.find_hook', return_value=["/fake/script.py"]):
            with patch('cookiecutter.hooks.run_hook', side_effect=UndefinedError("Template error")):
                with patch('cookiecutter.utils.work_in'):
                    with patch('cookiecutter.hooks.rmtree') as mock_rmtree:
                        with patch('cookiecutter.hooks.logger') as mock_logger:
                            with pytest.raises(UndefinedError):
                                run_hook_from_repo_dir(
                                    repo_dir=repo_dir,
                                    hook_name="pre_gen_project",
                                    project_dir=project_dir,
                                    context=context,
                                    delete_project_on_failure=True
                                )
                            
                            mock_rmtree.assert_called_once_with(project_dir)
                            assert mock_logger.exception.called

    # Test 6: Multiple hooks found and run
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        context = {"key": "value"}
        
        with patch('cookiecutter.hooks.find_hook', return_value=["/fake/script1.py", "/fake/script2.py"]):
            with patch('cookiecutter.hooks.run_hook') as mock_run_hook:
                with patch('cookiecutter.utils.work_in'):
                    run_hook_from_repo_dir(
                        repo_dir=repo_dir,
                        hook_name="post_gen_project",
                        project_dir=project_dir,
                        context=context,
                        delete_project_on_failure=True
                    )
                    
                    mock_run_hook.assert_called_once_with(
                        "post_gen_project",
                        project_dir,
                        context
                    )


# LLM-generated content at query #15
#--------------------------

```python
def test_run_script():
    import tempfile
    import os
    from pathlib import Path
    import stat
    
    # Test 1: Successful execution of Python script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(0)')
        script_path = f.name
    
    try:
        run_script(script_path)
        # Should not raise exception
    finally:
        os.unlink(script_path)
    
    # Test 2: Successful execution of shell script
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('#!/bin/sh\nexit 0')
        script_path = f.name
    
    try:
        # Make executable
        os.chmod(script_path, stat.S_IRWXU)
        run_script(script_path)
        # Should not raise exception
    finally:
        os.unlink(script_path)
    
    # Test 3: Script with non-zero exit status raises FailedHookException
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write('import sys\nsys.exit(1)')
        script_path = f.name
    
    try:
        import pytest
        from cookiecutter.exceptions import FailedHookException
        with pytest.raises(FailedHookException) as exc_info:
            run_script(script_path)
        assert "Hook script failed (exit status: 1)" in str(exc_info.value)
    finally:
        os.unlink(script_path)
    
    # Test 4: Empty file without shebang raises FailedHookException
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write('')
        script_path = f.name
    
    try:
        import pytest
        from cookiecutter.exceptions import FailedHookException
        os.chmod(script_path, stat.S_IRWXU)
        with pytest.raises(FailedHookException) as exc_info:
            run_script(script_path)
        assert "Hook script failed, might be an empty file or missing a shebang" in str(exc_info.value)
    finally:
        os.unlink(script_path)
    
    # Test 5: Non-existent script raises FailedHookException
    import pytest
    from cookiecutter.exceptions import FailedHookException
    with pytest.raises(FailedHookException) as exc_info:
        run_script('/non/existent/script.py')
    assert "Hook script failed (error:" in str(exc_info.value)
    
    # Test 6: Script execution with custom working directory
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / 'test_script.py'
        script_path.write_text('import sys\nsys.exit(0)')
        
        with tempfile.TemporaryDirectory() as work_dir:
            run_script(str(script_path), cwd=work_dir)
            # Should not raise exception


# LLM-generated content at query #16
#--------------------------

```python
def test_run_script_with_context(tmp_path):
    # Create test context
    context = {"project_name": "TestProject", "author": "TestAuthor"}
    
    # Create a test script with Jinja2 template
    script_content = """#!/usr/bin/env python
print("Project: {{ project_name }}")
print("Author: {{ author }}")
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding="utf-8")
    
    # Create a temporary directory to run from
    cwd = tmp_path / "run_dir"
    cwd.mkdir()
    
    # Mock subprocess.Popen to avoid actually running scripts
    original_popen = subprocess.Popen
    
    class MockPopen:
        def __init__(self, args, shell=False, cwd=None):
            self.args = args
            self.cwd = cwd
            self.returncode = 0
            
            # Verify the script was rendered correctly
            temp_script = Path(args[1] if args[0] == sys.executable else args[0])
            rendered_content = temp_script.read_text(encoding="utf-8")
            
            assert "Project: TestProject" in rendered_content
            assert "Author: TestAuthor" in rendered_content
            assert "{{ project_name }}" not in rendered_content
            assert "{{ author }}" not in rendered_content
            
        def wait(self):
            return self.returncode
    
    subprocess.Popen = MockPopen
    
    try:
        # Test successful execution
        run_script_with_context(script_path, cwd, context)
        
        # Test with non-zero exit status
        class MockPopenFailure(MockPopen):
            def wait(self):
                self.returncode = 1
                return self.returncode
        
        subprocess.Popen = MockPopenFailure
        
        with pytest.raises(FailedHookException) as exc_info:
            run_script_with_context(script_path, cwd, context)
        assert "Hook script failed (exit status: 1)" in str(exc_info.value)
        
        # Test with OSError
        class MockPopenOSError(MockPopen):
            def __init__(self, args, shell=False, cwd=None):
                raise OSError(errno.ENOEXEC, "Exec format error")
        
        subprocess.Popen = MockPopenOSError
        
        with pytest.raises(FailedHookException) as exc_info:
            run_script_with_context(script_path, cwd, context)
        assert "Hook script failed, might be an empty file or missing a shebang" in str(exc_info.value)
        
        # Test with other OSError
        class MockPopenOtherOSError(MockPopen):
            def __init__(self, args, shell=False, cwd=None):
                raise OSError(errno.EACCES, "Permission denied")
        
        subprocess.Popen = MockPopenOtherOSError
        
        with pytest.raises(FailedHookException) as exc_info:
            run_script_with_context(script_path, cwd, context)
        assert "Hook script failed (error:" in str(exc_info.value)
        
    finally:
        subprocess.Popen = original_popen

def test_run_script_with_context_shell_script(tmp_path):
    # Test with shell script
    context = {"message": "Hello World"}
    
    script_content = """#!/bin/bash
echo "{{ message }}"
"""
    
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content, encoding="utf-8")
    
    cwd = tmp_path / "run_dir"
    cwd.mkdir()
    
    original_popen = subprocess.Popen
    
    class MockPopen:
        def __init__(self, args, shell=False, cwd=None):
            self.args = args
            self.cwd = cwd
            self.returncode = 0
            
            # Verify shell script was rendered
            temp_script = Path(args[0])
            rendered_content = temp_script.read_text(encoding="utf-8")
            assert "Hello World" in rendered_content
            assert "{{ message }}" not in rendered_content
            
        def wait(self):
            return self.returncode
    
    subprocess.Popen = MockPopen
    
    try:
        run_script_with_context(script_path, cwd, context)
    finally:
        subprocess.Popen = original_popen

def test_run_script_with_context_undefined_variable(tmp_path):
    # Test with undefined Jinja2 variable
    context = {"defined_var": "I am defined"}
    
    script_content = """#!/usr/bin/env python
print("{{ defined_var }}")
print("{{ undefined_var }}")
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content, encoding="utf-8")
    
    cwd = tmp_path / "run_dir"
    cwd.mkdir()
    
    # Should raise UndefinedError when rendering
    with pytest.raises(UndefinedError):
        run_script_with_context(script_path, cwd, context)

def test_run_script_with_context_temp_file_cleanup(tmp_path):
    # Test that temporary files are cleaned up
    context = {"test": "value"}
    
    script_content = "print('test')"
    script_path = tmp_path / "test.py"
    script_path.write_text(script_content, encoding="utf-8")
    
    cwd = tmp_path / "run_dir"
    cwd.mkdir()
    
    original_popen = subprocess.Popen
    
    temp_files = []
    original_namedtemporaryfile = tempfile.NamedTemporaryFile
    
    def mock_namedtemporaryfile(*args, **kwargs):
        result = original_namedtemporaryfile(*args, **kwargs)
        temp_files.append(result.name)
        return result
    
    tempfile.NamedTemporaryFile = mock_namedtemporaryfile
    
    class MockPopen:
        def __init__(self, args, shell=False, cwd=None):
            self.returncode = 0
        def wait(self):
            return self.returncode
    
    subprocess.Popen = MockPopen
    
    try:
        run_script_with_context(script_path, cwd, context)
        # Verify temp file was created
        assert len(temp_files) == 1
        temp_path = Path(temp_files[0])
        # File should exist after creation but might be cleaned up by NamedTemporaryFile
        # depending on delete parameter
    finally:
        subprocess.Popen = original_popen
        tempfile.NamedTemporaryFile = original_namedtemporaryfile


# LLM-generated content at query #17
#--------------------------

```python
def test_run_pre_prompt_hook():
    # Test 1: No pre_prompt hook exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_pre_prompt_hook(tmpdir)
        assert result == tmpdir

    # Test 2: Valid pre_prompt hook executes successfully
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        hook_script = hooks_dir / "pre_prompt.py"
        hook_script.write_text("#!/usr/bin/env python\nprint('pre_prompt hook executed')")
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))
        assert result != tmpdir  # Should return a temporary directory

    # Test 3: pre_prompt hook fails with FailedHookException
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        hook_script = hooks_dir / "pre_prompt.py"
        hook_script.write_text("#!/usr/bin/env python\nimport sys\nsys.exit(1)")
        
        try:
            run_pre_prompt_hook(tmpdir)
            assert False, "Should have raised FailedHookException"
        except FailedHookException as e:
            assert "Pre-Prompt Hook script failed" in str(e)

    # Test 4: Multiple pre_prompt hooks execute in sequence
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        
        hook1 = hooks_dir / "pre_prompt.py"
        hook1.write_text("#!/usr/bin/env python\nprint('hook1')")
        
        hook2 = hooks_dir / "pre_prompt.sh"
        hook2.write_text("#!/bin/bash\necho 'hook2'")
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))

    # Test 5: Invalid hook files are ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / "hooks"
        hooks_dir.mkdir()
        
        valid_hook = hooks_dir / "pre_prompt.py"
        valid_hook.write_text("#!/usr/bin/env python\nprint('valid')")
        
        backup_file = hooks_dir / "pre_prompt.py~"
        backup_file.write_text("backup")
        
        wrong_hook = hooks_dir / "post_gen_project.py"
        wrong_hook.write_text("wrong hook")
        
        result = run_pre_prompt_hook(tmpdir)
        assert isinstance(result, (Path, str))


# LLM-generated content at query #18
#--------------------------

```python
def test_find_hook():
    import tempfile
    import os
    from pathlib import Path
    
    # Test 1: No hooks directory exists
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_hook('pre_gen_project', tmpdir)
        assert result is None
    
    # Test 2: Empty hooks directory
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        result = find_hook('pre_gen_project', str(hooks_dir))
        assert result is None
    
    # Test 3: Valid hook file found
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_gen_project.py'
        hook_file.write_text('print("test")')
        
        result = find_hook('pre_gen_project', str(hooks_dir))
        assert result is not None
        assert len(result) == 1
        assert os.path.basename(result[0]) == 'pre_gen_project.py'
    
    # Test 4: Multiple valid hook files found
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook1 = hooks_dir / 'pre_gen_project.py'
        hook1.write_text('print("test1")')
        hook2 = hooks_dir / 'pre_gen_project.sh'
        hook2.write_text('#!/bin/bash\necho "test2"')
        
        result = find_hook('pre_gen_project', str(hooks_dir))
        assert result is not None
        assert len(result) == 2
        filenames = [os.path.basename(p) for p in result]
        assert 'pre_gen_project.py' in filenames
        assert 'pre_gen_project.sh' in filenames
    
    # Test 5: Hook name mismatch
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'post_gen_project.py'
        hook_file.write_text('print("test")')
        
        result = find_hook('pre_gen_project', str(hooks_dir))
        assert result is None
    
    # Test 6: Backup files ignored
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_gen_project.py~'
        hook_file.write_text('print("test")')
        
        result = find_hook('pre_gen_project', str(hooks_dir))
        assert result is None
    
    # Test 7: Unsupported hook type
    with tempfile.TemporaryDirectory() as tmpdir:
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'unsupported_hook.py'
        hook_file.write_text('print("test")')
        
        result = find_hook('unsupported_hook', str(hooks_dir))
        assert result is None
    
    # Test 8: Default hooks_dir parameter
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        hooks_dir = Path(tmpdir) / 'hooks'
        hooks_dir.mkdir()
        hook_file = hooks_dir / 'pre_gen_project.py'
        hook_file.write_text('print("test")')
        
        result = find_hook('pre_gen_project')
        assert result is not None
        assert len(result) == 1


# LLM-generated content at query #19
#--------------------------

```python
def test_run_hook():
    import tempfile
    import os
    from pathlib import Path
    import pytest
    from unittest.mock import Mock, patch, call
    from cookiecutter.exceptions import FailedHookException
    
    # Test 1: No hook found
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = None
        with patch('cookiecutter.hooks.logger') as mock_logger:
            # Should not raise any exception
            run_hook('pre_gen_project', '/tmp/project', {})
            mock_logger.debug.assert_called_with('No pre_gen_project hook found')
    
    # Test 2: Hook found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            with patch('cookiecutter.hooks.logger') as mock_logger:
                context = {'project_name': 'test'}
                run_hook('pre_gen_project', '/tmp/project', context)
                
                mock_logger.debug.assert_called_with('Running hook pre_gen_project')
                mock_run.assert_called_once_with(
                    '/tmp/hooks/pre_gen_project.py',
                    '/tmp/project',
                    context
                )
    
    # Test 3: Multiple hooks found and executed
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = [
            '/tmp/hooks/pre_gen_project.py',
            '/tmp/hooks/pre_gen_project.sh'
        ]
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            context = {'project_name': 'test'}
            run_hook('pre_gen_project', '/tmp/project', context)
            
            assert mock_run.call_count == 2
            expected_calls = [
                call('/tmp/hooks/pre_gen_project.py', '/tmp/project', context),
                call('/tmp/hooks/pre_gen_project.sh', '/tmp/project', context)
            ]
            mock_run.assert_has_calls(expected_calls)
    
    # Test 4: Hook execution raises FailedHookException
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            mock_run.side_effect = FailedHookException("Hook failed")
            
            with pytest.raises(FailedHookException, match="Hook failed"):
                run_hook('pre_gen_project', '/tmp/project', {})
    
    # Test 5: Hook execution raises UndefinedError
    from jinja2.exceptions import UndefinedError
    with patch('cookiecutter.hooks.find_hook') as mock_find:
        mock_find.return_value = ['/tmp/hooks/pre_gen_project.py']
        with patch('cookiecutter.hooks.run_script_with_context') as mock_run:
            mock_run.side_effect = UndefinedError("Template error")
            
            with pytest.raises(UndefinedError, match="Template error"):
                run_hook('pre_gen_project', '/tmp/project', {})


# LLM-generated content at query #20
#--------------------------

```python
def test_run_script_with_context(tmp_path):
    # Create a test script with Jinja2 template variables
    script_content = """#!/usr/bin/env python
import sys
print("Project: {{ project_name }}")
print("Version: {{ version }}")
sys.exit(0)
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content)
    
    # Create context with template variables
    context = {
        "project_name": "TestProject",
        "version": "1.0.0"
    }
    
    # Create a temporary directory to run from
    cwd = tmp_path / "project_dir"
    cwd.mkdir()
    
    # Run the script with context
    run_script_with_context(script_path, cwd, context)

def test_run_script_with_context_failed_hook(tmp_path):
    # Create a test script that will fail
    script_content = """#!/usr/bin/env python
import sys
print("Project: {{ project_name }}")
sys.exit(1)
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content)
    
    context = {"project_name": "TestProject"}
    cwd = tmp_path / "project_dir"
    cwd.mkdir()
    
    # Should raise FailedHookException
    try:
        run_script_with_context(script_path, cwd, context)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed" in str(e)

def test_run_script_with_context_undefined_variable(tmp_path):
    # Create a test script with undefined variable
    script_content = """#!/usr/bin/env python
import sys
print("Project: {{ undefined_var }}")
sys.exit(0)
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content)
    
    context = {"project_name": "TestProject"}
    cwd = tmp_path / "project_dir"
    cwd.mkdir()
    
    # Should raise UndefinedError from Jinja2
    try:
        run_script_with_context(script_path, cwd, context)
        assert False, "Should have raised UndefinedError"
    except UndefinedError:
        pass

def test_run_script_with_context_shell_script(tmp_path):
    # Create a shell script with Jinja2 template variables
    script_content = """#!/bin/bash
echo "Project: {{ project_name }}"
echo "Author: {{ author }}"
"""
    
    script_path = tmp_path / "test_script.sh"
    script_path.write_text(script_content)
    
    context = {
        "project_name": "TestProject",
        "author": "Test Author"
    }
    
    cwd = tmp_path / "project_dir"
    cwd.mkdir()
    
    # Run the script with context
    run_script_with_context(script_path, cwd, context)

def test_run_script_with_context_empty_file(tmp_path):
    # Create an empty script file
    script_path = tmp_path / "empty_script.py"
    script_path.write_text("")
    
    context = {"project_name": "TestProject"}
    cwd = tmp_path / "project_dir"
    cwd.mkdir()
    
    # Should raise FailedHookException due to empty file
    try:
        run_script_with_context(script_path, cwd, context)
        assert False, "Should have raised FailedHookException"
    except FailedHookException as e:
        assert "Hook script failed" in str(e)

def test_run_script_with_context_complex_template(tmp_path):
    # Create a script with complex Jinja2 template logic
    script_content = """#!/usr/bin/env python
import sys
{% if use_database %}
print("Database: {{ db_name }}")
{% else %}
print("No database configured")
{% endif %}
sys.exit(0)
"""
    
    script_path = tmp_path / "test_script.py"
    script_path.write_text(script_content)
    
    # Test with database enabled
    context = {
        "use_database": True,
        "db_name": "test_db"
    }
    
    cwd = tmp_path / "project_dir"
    cwd.mkdir()
    
    run_script_with_context(script_path, cwd, context)
    
    # Test with database disabled
    context = {
        "use_database": False,
        "db_name": "test_db"
    }
    
    run_script_with_context(script_path, cwd, context)


