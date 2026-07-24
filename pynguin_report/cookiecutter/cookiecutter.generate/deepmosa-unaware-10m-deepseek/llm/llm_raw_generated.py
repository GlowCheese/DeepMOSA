####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generate_files():
    import tempfile
    import json
    from pathlib import Path
    from unittest.mock import patch, MagicMock, call
    import shutil
    import os
    
    # Test 1: Basic file generation with template rendering
    def test_basic_generation():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple template structure
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            # Create cookiecutter.json
            context_data = {
                "project_name": "TestProject",
                "version": "1.0.0"
            }
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump(context_data, f)
            
            # Create a template file
            template_file = repo_dir / "{{cookiecutter.project_name}}.txt"
            template_file.write_text("Version: {{cookiecutter.version}}")
            
            # Create output directory
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # Generate files
            context = {
                "cookiecutter": {
                    "project_name": "TestProject",
                    "version": "1.0.0",
                    "_new_lines": "\n"
                }
            }
            
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False,
                skip_if_file_exists=False,
                accept_hooks=False
            )
            
            # Verify the generated file
            expected_file = output_dir / "TestProject" / "TestProject.txt"
            assert expected_file.exists()
            assert expected_file.read_text() == "Version: 1.0.0"
    
    # Test 2: Overwrite existing directory
    def test_overwrite_if_exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump({"name": "Test"}, f)
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("Content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # First generation
            context = {"cookiecutter": {"name": "Test", "_new_lines": "\n"}}
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False,
                accept_hooks=False
            )
            
            # Modify the template
            template_file.write_text("Updated Content")
            
            # Try to generate again without overwrite - should raise exception
            try:
                generate_files(
                    repo_dir=str(repo_dir),
                    context=context,
                    output_dir=str(output_dir),
                    overwrite_if_exists=False,
                    accept_hooks=False
                )
                assert False, "Should have raised OutputDirExistsException"
            except OutputDirExistsException:
                pass
            
            # Generate with overwrite - should succeed
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=True,
                accept_hooks=False
            )
            
            expected_file = output_dir / "template" / "test.txt"
            assert expected_file.read_text() == "Updated Content"
    
    # Test 3: Skip if file exists
    def test_skip_if_file_exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump({"name": "Test"}, f)
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("Original")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # First generation
            context = {"cookiecutter": {"name": "Test", "_new_lines": "\n"}}
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                skip_if_file_exists=False,
                accept_hooks=False
            )
            
            # Modify the generated file
            generated_file = output_dir / "template" / "test.txt"
            generated_file.write_text("Modified")
            
            # Generate again with skip_if_file_exists=True
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                skip_if_file_exists=True,
                accept_hooks=False
            )
            
            # File should still have the modified content
            assert generated_file.read_text() == "Modified"
    
    # Test 4: Copy without render
    def test_copy_without_render():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_data = {
                "name": "Test",
                "_copy_without_render": ["*.bin", "data/*"]
            }
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump(context_data, f)
            
            # Create files that should be copied without rendering
            binary_file = repo_dir / "test.bin"
            binary_file.write_bytes(b"binary content")
            
            data_dir = repo_dir / "data"
            data_dir.mkdir()
            data_file = data_dir / "info.txt"
            data_file.write_text("{{should_not_render}}")
            
            # Create a file that should be rendered
            template_file = repo_dir / "{{cookiecutter.name}}.txt"
            template_file.write_text("Name: {{cookiecutter.name}}")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {
                "cookiecutter": {
                    "name": "TestProject",
                    "_copy_without_render": ["*.bin", "data/*"],
                    "_new_lines": "\n"
                }
            }
            
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                accept_hooks=False
            )
            
            # Verify binary file was copied without rendering
            copied_binary = output_dir / "template" / "test.bin"
            assert copied_binary.exists()
            assert copied_binary.read_bytes() == b"binary content"
            
            # Verify data file was copied without rendering
            copied_data = output_dir / "template" / "data" / "info.txt"
            assert copied_data.exists()
            assert copied_data.read_text() == "{{should_not_render}}"
            
            # Verify template file was rendered
            rendered_file = output_dir / "template" / "TestProject.txt"
            assert rendered_file.exists()
            assert rendered_file.read_text() == "Name: TestProject"
    
    # Test 5: Hooks execution
    def test_hooks_execution():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump({"name": "Test"}, f)
            
            # Create hooks directory and pre-gen hook
            hooks_dir = repo_dir / "hooks"
            hooks_dir.mkdir()
            pre_hook = hooks_dir / "pre_gen_project.py"
            pre_hook.write_text("""
import os
with open('pre_hook_ran.txt', 'w') as f:
    f.write('pre hook executed')
""")
            
            post_hook = hooks_dir / "post_gen_project.py"
            post_hook.write_text("""
import os
with open('post_hook_ran.txt', 'w') as f:
    f.write('post hook executed')
""")
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("Content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {"cookiecutter": {"name": "Test", "_new_lines": "\n"}}
            
            # Mock run_hook_from_repo_dir to track calls
            with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_hook:
                generate_files(
                    repo_dir=str(repo_dir),
                    context=context,
                    output_dir=str(output_dir),
                    accept_hooks=True
                )
                
                # Verify hooks were called
                assert mock_hook.call_count == 2
                calls = mock_hook.call_args_list
                
                # First call should be pre_gen_project
                assert calls[0][0][1] == 'pre_gen_project'
                
                # Second call should be post_gen_project
                assert calls[1][0][1] == 'post_gen_project'
    
    # Test 6: Undefined variable error
    def test_undefined_variable():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump({"name": "Test"}, f)
            
            # Create template with undefined variable
            template_file = repo_dir /


# LLM-generated content at query #2
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test 1: Simple overwrite of existing variable
    context = {"name": "old", "version": "1.0"}
    overwrite = {"name": "new"}
    apply_overwrites_to_context(context, overwrite)
    assert context["name"] == "new"
    assert context["version"] == "1.0"

    # Test 2: New variable on first level should be ignored
    context = {"existing": "value"}
    overwrite = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite)
    assert "new_var" not in context
    assert context["existing"] == "value"

    # Test 3: Multichoice variable - valid subset
    context = {"choices": ["a", "b", "c", "d"]}
    overwrite = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite)
    assert context["choices"] == ["b", "c"]

    # Test 4: Multichoice variable - invalid subset should raise ValueError
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["b", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provided for multi-choice variable" in str(e)

    # Test 5: Choice variable - valid choice becomes first
    context = {"choice": ["first", "second", "third"]}
    overwrite = {"choice": "second"}
    apply_overwrites_to_context(context, overwrite)
    assert context["choice"] == ["second", "first", "third"]

    # Test 6: Choice variable - invalid choice should raise ValueError
    context = {"choice": ["a", "b", "c"]}
    overwrite = {"choice": "d"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provided for choice variable" in str(e)

    # Test 7: Dictionary variable - partial overwrite
    context = {"config": {"key1": "val1", "key2": "val2", "key3": "val3"}}
    overwrite = {"config": {"key2": "new_val2", "key4": "val4"}}
    apply_overwrites_to_context(context, overwrite)
    assert context["config"]["key1"] == "val1"
    assert context["config"]["key2"] == "new_val2"
    assert context["config"]["key3"] == "val3"
    assert context["config"]["key4"] == "val4"

    # Test 8: Boolean variable - string "yes" converts to True
    context = {"flag": False}
    overwrite = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite)
    assert context["flag"] is True

    # Test 9: Boolean variable - string "no" converts to False
    context = {"flag": True}
    overwrite = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite)
    assert context["flag"] is False

    # Test 10: Boolean variable - invalid string should raise ValueError
    context = {"flag": True}
    overwrite = {"flag": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)

    # Test 11: Nested dictionary in dictionary variable
    context = {"deep": {"level1": {"level2": "old"}}}
    overwrite = {"deep": {"level1": {"level2": "new", "level2_new": "added"}}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context["deep"]["level1"]["level2"] == "new"
    assert context["deep"]["level1"]["level2_new"] == "added"

    # Test 12: New variable in deeper level when in_dictionary_variable=True
    context = {"existing": {"nested": "value"}}
    overwrite = {"new_key": "new_value"}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context["new_key"] == "new_value"


# LLM-generated content at query #3
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test 1: Simple overwrite of existing variable
    context = {"name": "old_value", "version": "1.0"}
    overwrite = {"name": "new_value"}
    apply_overwrites_to_context(context, overwrite)
    assert context["name"] == "new_value"
    assert context["version"] == "1.0"

    # Test 2: New variable on first level should be ignored
    context = {"existing": "value"}
    overwrite = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite)
    assert "new_var" not in context
    assert context["existing"] == "value"

    # Test 3: Overwrite list variable with valid choice
    context = {"choice_var": ["option1", "option2", "option3"]}
    overwrite = {"choice_var": "option2"}
    apply_overwrites_to_context(context, overwrite)
    assert context["choice_var"] == ["option2", "option1", "option3"]

    # Test 4: Overwrite list variable with invalid choice raises ValueError
    context = {"choice_var": ["option1", "option2"]}
    overwrite = {"choice_var": "invalid_option"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid_option" in str(e)
        assert "choice variable" in str(e)

    # Test 5: Overwrite multichoice variable with valid subset
    context = {"multichoice": ["a", "b", "c", "d"]}
    overwrite = {"multichoice": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite)
    assert context["multichoice"] == ["b", "c"]

    # Test 6: Overwrite multichoice variable with invalid subset raises ValueError
    context = {"multichoice": ["a", "b", "c"]}
    overwrite = {"multichoice": ["a", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "multi-choice variable" in str(e)

    # Test 7: Partial overwrite of nested dictionary
    context = {"nested": {"key1": "value1", "key2": "value2", "key3": "value3"}}
    overwrite = {"nested": {"key2": "updated", "key4": "new_value"}}
    apply_overwrites_to_context(context, overwrite)
    assert context["nested"]["key1"] == "value1"
    assert context["nested"]["key2"] == "updated"
    assert context["nested"]["key3"] == "value3"
    assert context["nested"]["key4"] == "new_value"

    # Test 8: Overwrite boolean variable with string "yes"
    context = {"flag": False}
    overwrite = {"flag": "yes"}
    apply_overwrites_to_context(context, overwrite)
    assert context["flag"] is True

    # Test 9: Overwrite boolean variable with string "no"
    context = {"flag": True}
    overwrite = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite)
    assert context["flag"] is False

    # Test 10: Overwrite boolean variable with invalid string raises ValueError
    context = {"flag": True}
    overwrite = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)

    # Test 11: New variable in nested dictionary should be added
    context = {"nested": {"existing": "value"}}
    overwrite = {"nested": {"new_key": "new_value"}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context["nested"]["existing"] == "value"
    assert context["nested"]["new_key"] == "new_value"

    # Test 12: Complex nested structure
    context = {
        "top": "value",
        "config": {
            "enabled": True,
            "choices": ["opt1", "opt2", "opt3"],
            "settings": {"timeout": 30, "retries": 3}
        }
    }
    overwrite = {
        "top": "updated",
        "config": {
            "enabled": "no",
            "choices": "opt2",
            "settings": {"timeout": 60, "new_setting": "added"}
        }
    }
    apply_overwrites_to_context(context, overwrite)
    assert context["top"] == "updated"
    assert context["config"]["enabled"] is False
    assert context["config"]["choices"] == ["opt2", "opt1", "opt3"]
    assert context["config"]["settings"]["timeout"] == 60
    assert context["config"]["settings"]["retries"] == 3
    assert context["config"]["settings"]["new_setting"] == "added"


# LLM-generated content at query #4
#--------------------------

```python
def test_render_and_create_dir():
    import tempfile
    import shutil
    from pathlib import Path
    from jinja2 import Environment
    
    # Test 1: Normal directory creation with rendered name
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {'project_name': 'MyProject'}
        environment = Environment()
        
        dirname = "{{ project_name }}_app"
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, environment
        )
        
        expected_dir = output_dir / "MyProject_app"
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
        assert expected_dir.is_dir()
    
    # Test 2: Directory already exists without overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {'project_name': 'ExistingProject'}
        environment = Environment()
        
        dirname = "{{ project_name }}"
        existing_dir = output_dir / "ExistingProject"
        existing_dir.mkdir()
        
        try:
            render_and_create_dir(dirname, context, output_dir, environment)
            assert False, "Should have raised OutputDirExistsException"
        except OutputDirExistsException as e:
            assert "already exists" in str(e)
    
    # Test 3: Directory already exists with overwrite
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {'project_name': 'ExistingProject'}
        environment = Environment()
        
        dirname = "{{ project_name }}"
        existing_dir = output_dir / "ExistingProject"
        existing_dir.mkdir()
        
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, environment, overwrite_if_exists=True
        )
        
        assert result_dir == existing_dir
        assert created is False
        assert existing_dir.exists()
    
    # Test 4: Empty directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {}
        environment = Environment()
        
        try:
            render_and_create_dir("", context, output_dir, environment)
            assert False, "Should have raised EmptyDirNameException"
        except EmptyDirNameException as e:
            assert "directory name is empty" in str(e)
    
    # Test 5: Complex template rendering
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {
            'author': 'John Doe',
            'year': '2023',
            'version': '1.0'
        }
        environment = Environment()
        
        dirname = "project_{{ author }}_{{ year }}_{{ version }}"
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, environment
        )
        
        expected_dir = output_dir / "project_John Doe_2023_1.0"
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
    
    # Test 6: Nested directory path
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {'module': 'utils'}
        environment = Environment()
        
        dirname = "src/{{ module }}/tests"
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, environment
        )
        
        expected_dir = output_dir / "src/utils/tests"
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
        assert (output_dir / "src").exists()
        assert (output_dir / "src/utils").exists()
    
    # Test 7: Template with filters
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        context = {'name': 'my-project'}
        environment = Environment()
        
        dirname = "{{ name|upper|replace('-', '_') }}"
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, environment
        )
        
        expected_dir = output_dir / "MY_PROJECT"
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_files():
    import tempfile
    import json
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import pytest
    from cookiecutter.exceptions import (
        UndefinedVariableInTemplate,
        OutputDirExistsException,
    )

    # Test 1: Basic file generation
    def test_basic_generation():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create template structure
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            # Create cookiecutter.json
            context_data = {"project_name": "TestProject", "version": "1.0"}
            context_file = repo_dir / "cookiecutter.json"
            context_file.write_text(json.dumps(context_data))
            
            # Create a template file
            template_file = repo_dir / "{{cookiecutter.project_name}}.txt"
            template_file.write_text("Version: {{cookiecutter.version}}")
            
            # Create output directory
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # Generate files
            context = {"cookiecutter": context_data}
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False,
                skip_if_file_exists=False,
                accept_hooks=False,
                keep_project_on_failure=False
            )
            
            # Verify results
            expected_file = output_dir / "TestProject" / "TestProject.txt"
            assert expected_file.exists()
            assert expected_file.read_text() == "Version: 1.0"
            assert Path(result) == expected_file.parent

    # Test 2: Overwrite existing directory
    def test_overwrite_existing():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_file = repo_dir / "cookiecutter.json"
            context_file.write_text(json.dumps({"name": "Test"}))
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # Create existing project directory
            existing_dir = output_dir / "Test"
            existing_dir.mkdir()
            existing_file = existing_dir / "existing.txt"
            existing_file.write_text("old content")
            
            # Should raise exception without overwrite
            with pytest.raises(OutputDirExistsException):
                generate_files(
                    repo_dir=str(repo_dir),
                    output_dir=str(output_dir),
                    overwrite_if_exists=False
                )
            
            # Should succeed with overwrite
            result = generate_files(
                repo_dir=str(repo_dir),
                output_dir=str(output_dir),
                overwrite_if_exists=True
            )
            
            # Old content should be gone
            assert not existing_file.exists()
            assert (Path(result) / "test.txt").exists()

    # Test 3: Skip existing files
    def test_skip_existing_files():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_file = repo_dir / "cookiecutter.json"
            context_file.write_text(json.dumps({"name": "Test"}))
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("new content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # Create existing file in output
            existing_dir = output_dir / "Test"
            existing_dir.mkdir()
            existing_file = existing_dir / "test.txt"
            existing_file.write_text("old content")
            
            # Generate with skip_if_file_exists=True
            result = generate_files(
                repo_dir=str(repo_dir),
                output_dir=str(output_dir),
                overwrite_if_exists=True,
                skip_if_file_exists=True
            )
            
            # File should not be overwritten
            assert existing_file.read_text() == "old content"

    # Test 4: Copy without render
    def test_copy_without_render():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_data = {
                "name": "Test",
                "_copy_without_render": ["*.bin", "data/*"]
            }
            context_file = repo_dir / "cookiecutter.json"
            context_file.write_text(json.dumps(context_data))
            
            # Create files that should not be rendered
            binary_file = repo_dir / "test.bin"
            binary_file.write_bytes(b"binary\x00content")
            
            data_dir = repo_dir / "data"
            data_dir.mkdir()
            data_file = data_dir / "info.dat"
            data_file.write_bytes(b"data\x00file")
            
            # Create template file that should be rendered
            template_file = repo_dir / "{{cookiecutter.name}}.txt"
            template_file.write_text("Name: {{cookiecutter.name}}")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {"cookiecutter": context_data}
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                accept_hooks=False
            )
            
            # Verify binary file was copied without rendering
            output_binary = Path(result) / "test.bin"
            assert output_binary.exists()
            assert output_binary.read_bytes() == b"binary\x00content"
            
            # Verify data file was copied
            output_data = Path(result) / "data" / "info.dat"
            assert output_data.exists()
            
            # Verify template was rendered
            output_template = Path(result) / "Test.txt"
            assert output_template.exists()
            assert output_template.read_text() == "Name: Test"

    # Test 5: Undefined variable error
    def test_undefined_variable():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_file = repo_dir / "cookiecutter.json"
            context_file.write_text(json.dumps({"name": "Test"}))
            
            # Template with undefined variable
            template_file = repo_dir / "{{undefined_var}}.txt"
            template_file.write_text("content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            with pytest.raises(UndefinedVariableInTemplate):
                generate_files(
                    repo_dir=str(repo_dir),
                    output_dir=str(output_dir),
                    accept_hooks=False
                )

    # Test 6: With hooks
    def test_with_hooks():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_file = repo_dir / "cookiecutter.json"
            context_file.write_text(json.dumps({"name": "Test"}))
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("content")
            
            # Create hooks directory and pre-gen hook
            hooks_dir = repo_dir / "hooks"
            hooks_dir.mkdir()
            pre_hook = hooks_dir / "pre_gen_project.py"
            pre_hook.write_text("#!/usr/bin/env python\nprint('pre hook')")
            pre_hook.chmod(0o755)
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            with patch('cookiecutter.hooks.run_hook_from_repo_dir') as mock_hook:
                generate_files(
                    repo_dir=str(repo_dir),
                    output_dir=str(output_dir),
                    accept_hooks=True
                )
                
                # Verify hooks were called
                assert mock_hook.call_count == 2
                mock_hook.assert_any_call(
                    str(repo_dir),
                    'pre_gen_project',
                    str(Path(output_dir) / "Test"),
                    {'cookiecutter': {'name': 'Test'}},
                    True
                )

    # Test 7: Keep project on failure
    def test_keep_project_on_failure():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_file = repo_dir / "cookiecutter.json"
            context_file.write_text(json.dumps({"name": "Test"}))
            
            # Create a problematic template
            template_file = repo_dir / "{{bad_template}.txt"
            template_file.write_text("content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            try:
                generate_files(
                    repo_dir=str(repo_dir),
                    output_dir=str(output_dir),
                    accept_hooks=False,
                    keep_project_on_failure=True
                )
            except UndefinedVariableInTemplate:
                # Project directory should still exist
                project_dir = output_dir / "Test"
                assert project_dir.exists()

    # Run all tests
    test_basic_generation()
    test_overwrite_existing()
    test_skip_existing_files()
    test_copy_without_render()
    test_undefined_variable()
    test_with_hooks()
    test_keep_project_on_failure()


# LLM-generated content at query #6
#--------------------------

```python
def test_generate_files():
    import tempfile
    import shutil
    from pathlib import Path
    from collections import OrderedDict
    
    # Test 1: Basic file generation with simple template
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create a simple template file
        template_file = repo_dir / "{{ project_name }}.txt"
        template_file.write_text("Hello {{ name }}!")
        
        # Create cookiecutter.json
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text('{"project_name": "test_project", "name": "World"}')
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        context = generate_context(str(context_file))
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=False
        )
        
        generated_file = output_dir / "test_project" / "test_project.txt"
        assert generated_file.exists()
        assert generated_file.read_text() == "Hello World!"
    
    # Test 2: Test with _copy_without_render
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create files
        (repo_dir / "render.txt").write_text("Render {{ var }}")
        (repo_dir / "copy.txt").write_text("Copy {{ var }}")
        
        # Create cookiecutter.json with _copy_without_render
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(
            '{"project_name": "test", "var": "value", "_copy_without_render": ["copy.txt"]}'
        )
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        context = generate_context(str(context_file))
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir)
        )
        
        render_file = Path(result) / "render.txt"
        copy_file = Path(result) / "copy.txt"
        
        assert render_file.read_text() == "Render value"
        assert copy_file.read_text() == "Copy {{ var }"
    
    # Test 3: Test overwrite_if_exists behavior
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        template_file = repo_dir / "test.txt"
        template_file.write_text("Content")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text('{"project_name": "project"}')
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        context = generate_context(str(context_file))
        
        # First generation should succeed
        result1 = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=False
        )
        
        # Second generation without overwrite should fail
        try:
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False
            )
            assert False, "Should have raised OutputDirExistsException"
        except OutputDirExistsException:
            pass  # Expected
        
        # Third generation with overwrite should succeed
        result3 = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            overwrite_if_exists=True
        )
        
        assert Path(result3).exists()
    
    # Test 4: Test skip_if_file_exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        template_file = repo_dir / "test.txt"
        template_file.write_text("Original")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text('{"project_name": "project"}')
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        context = generate_context(str(context_file))
        
        # Create a file manually in the output directory
        manual_dir = output_dir / "project"
        manual_dir.mkdir()
        manual_file = manual_dir / "test.txt"
        manual_file.write_text("Manual content")
        
        # Generate with skip_if_file_exists=True
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            skip_if_file_exists=True
        )
        
        # File should not be overwritten
        assert Path(result).exists()
        existing_file = Path(result) / "test.txt"
        assert existing_file.read_text() == "Manual content"
    
    # Test 5: Test with binary files
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create a binary file
        binary_file = repo_dir / "binary.dat"
        binary_file.write_bytes(b'\x00\x01\x02\x03\x04')
        
        # Create a text file
        text_file = repo_dir / "text.txt"
        text_file.write_text("Text {{ var }}")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text('{"project_name": "project", "var": "rendered"}')
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        context = generate_context(str(context_file))
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir)
        )
        
        result_binary = Path(result) / "binary.dat"
        result_text = Path(result) / "text.txt"
        
        assert result_binary.read_bytes() == b'\x00\x01\x02\x03\x04'
        assert result_text.read_text() == "Text rendered"
    
    # Test 6: Test with directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create nested directory structure
        subdir = repo_dir / "subdir"
        subdir.mkdir()
        nested_subdir = subdir / "nested"
        nested_subdir.mkdir()
        
        (repo_dir / "root.txt").write_text("Root")
        (subdir / "sub.txt").write_text("Sub {{ var }}")
        (nested_subdir / "nested.txt").write_text("Nested")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text('{"project_name": "project", "var": "value"}')
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        context = generate_context(str(context_file))
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir)
        )
        
        result_path = Path(result)
        assert (result_path / "root.txt").exists()
        assert (result_path / "subdir" / "sub.txt").read_text() == "Sub value"
        assert (result_path / "subdir" / "nested" / "nested.txt").exists()
    
    # Test 7: Test with accept_hooks=False
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create hooks directory and a dummy hook
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        pre_hook = hooks_dir / "pre_gen_project.py"
        pre_hook.write_text("#!/usr/bin/env python\nprint('pre hook')")
        pre_hook.chmod(0o755)
        
        template_file = repo_dir / "test.txt"
        template_file.write_text("Content")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text('{"project_name": "project"}')
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        context = generate_context(str(context_file))
        
        # Should run without errors even with hooks present but accept_hooks=False
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=str(output_dir),
            accept_hooks=False
        )
        
        assert Path(result).exists()
    
    # Test 8: Test with empty context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        template_file = repo_dir / "simple.txt"
        template_file.write_text("Simple content")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text('{"project_name": "project"}')
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        # Pass None as context, should be generated from file
        result = generate_files(
            repo_dir=str(repo_dir),
            context=None,
            output_dir=str(output_dir)
        )
        
        assert Path(result).exists()
       


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context():
    import json
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.generate import generate_context, ContextDecodingException

    # Test 1: Normal context generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0.0"}, f)
        context_file = f.name

    try:
        context = generate_context(context_file)
        assert 'cookiecutter' in context
        assert context['cookiecutter']['project_name'] == "Test Project"
        assert context['cookiecutter']['version'] == "1.0.0"
    finally:
        os.unlink(context_file)

    # Test 2: With default_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": ["A", "B", "C"], "version": "1.0.0"}, f)
        context_file = f.name

    try:
        default_context = {"project_name": "B"}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['project_name'] == ["B", "A", "C"]
    finally:
        os.unlink(context_file)

    # Test 3: With extra_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0.0"}, f)
        context_file = f.name

    try:
        extra_context = {"project_name": "Overridden"}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project_name'] == "Overridden"
    finally:
        os.unlink(context_file)

    # Test 4: Invalid JSON raises ContextDecodingException
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": json}')
        context_file = f.name

    try:
        try:
            generate_context(context_file)
            assert False, "Should have raised ContextDecodingException"
        except ContextDecodingException:
            pass
    finally:
        os.unlink(context_file)

    # Test 5: Complex nested structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "project": {
                "name": "Test",
                "settings": {"debug": True}
            },
            "list": ["a", "b", "c"]
        }, f)
        context_file = f.name

    try:
        context = generate_context(context_file)
        assert context['cookiecutter']['project']['name'] == "Test"
        assert context['cookiecutter']['project']['settings']['debug'] is True
        assert context['cookiecutter']['list'] == ["a", "b", "c"]
    finally:
        os.unlink(context_file)

    # Test 6: Empty JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({}, f)
        context_file = f.name

    try:
        context = generate_context(context_file)
        assert context['cookiecutter'] == {}
    finally:
        os.unlink(context_file)

    # Test 7: Default context with invalid value shows warning
    import warnings
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"choice": ["A", "B"]}, f)
        context_file = f.name

    try:
        default_context = {"choice": "C"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            context = generate_context(context_file, default_context=default_context)
            assert len(w) == 1
            assert "Invalid default received" in str(w[0].message)
            assert context['cookiecutter']['choice'] == ["A", "B"]
    finally:
        os.unlink(context_file)


# LLM-generated content at query #8
#--------------------------

```python
def test_generate_files():
    import tempfile
    import shutil
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import json
    
    # Test 1: Basic file generation with template rendering
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create template files
        template_file = repo_dir / "{{project_name}}.txt"
        template_file.write_text("Hello {{author}}!")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({
            "project_name": "test_project",
            "author": "Test Author"
        }))
        
        context = generate_context(str(context_file))
        
        # Generate files
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=tmpdir
        )
        
        # Verify generated file
        generated_file = Path(tmpdir) / "test_project" / "test_project.txt"
        assert generated_file.exists()
        assert generated_file.read_text() == "Hello Test Author!"
    
    # Test 2: Copy-only files (without rendering)
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create files
        regular_file = repo_dir / "regular.txt"
        regular_file.write_text("Render {{variable}}")
        
        copy_file = repo_dir / "copy.txt"
        copy_file.write_text("Don't render {{variable}}")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({
            "project_name": "test",
            "_copy_without_render": ["copy.txt"]
        }))
        
        context = generate_context(str(context_file))
        
        # Generate files
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=tmpdir
        )
        
        # Verify files
        generated_dir = Path(tmpdir) / "test"
        regular_content = (generated_dir / "regular.txt").read_text()
        copy_content = (generated_dir / "copy.txt").read_text()
        
        assert regular_content == "Render "
        assert copy_content == "Don't render {{variable}}"
    
    # Test 3: Overwrite existing directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        template_file = repo_dir / "test.txt"
        template_file.write_text("content")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"project_name": "myproject"}))
        
        context = generate_context(str(context_file))
        
        # Create existing directory
        existing_dir = Path(tmpdir) / "myproject"
        existing_dir.mkdir()
        existing_file = existing_dir / "existing.txt"
        existing_file.write_text("old content")
        
        # Should raise exception without overwrite_if_exists
        try:
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=tmpdir,
                overwrite_if_exists=False
            )
            assert False, "Should have raised OutputDirExistsException"
        except OutputDirExistsException:
            pass
        
        # Should succeed with overwrite_if_exists
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=tmpdir,
            overwrite_if_exists=True
        )
        
        assert (Path(tmpdir) / "myproject" / "test.txt").exists()
    
    # Test 4: Skip if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        template_file = repo_dir / "test.txt"
        template_file.write_text("new content")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"project_name": "project"}))
        
        context = generate_context(str(context_file))
        
        # Create existing file in output
        output_dir = Path(tmpdir) / "project"
        output_dir.mkdir()
        existing_file = output_dir / "test.txt"
        existing_file.write_text("existing content")
        
        # Generate with skip_if_file_exists=True
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=tmpdir,
            skip_if_file_exists=True
        )
        
        # File should not be overwritten
        assert existing_file.read_text() == "existing content"
    
    # Test 5: Binary file handling
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create a binary file (PNG header)
        binary_file = repo_dir / "image.png"
        binary_file.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"project_name": "test"}))
        
        context = generate_context(str(context_file))
        
        # Generate files
        result = generate_files(
            repo_dir=str(repo_dir),
            context=context,
            output_dir=tmpdir
        )
        
        # Verify binary file was copied (not rendered)
        generated_binary = Path(tmpdir) / "test" / "image.png"
        assert generated_binary.exists()
        assert generated_binary.read_bytes() == binary_file.read_bytes()
    
    # Test 6: Hooks execution
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create template
        template_file = repo_dir / "test.txt"
        template_file.write_text("test")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"project_name": "hook_test"}))
        
        # Create hook directory and script
        hooks_dir = repo_dir / "hooks"
        hooks_dir.mkdir()
        pre_hook = hooks_dir / "pre_gen_project.py"
        pre_hook.write_text("#!/usr/bin/env python\nprint('pre hook executed')")
        pre_hook.chmod(0o755)
        
        context = generate_context(str(context_file))
        
        # Mock run_hook_from_repo_dir to verify it's called
        with patch('cookiecutter.generate.run_hook_from_repo_dir') as mock_hook:
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=tmpdir,
                accept_hooks=True
            )
            
            # Verify hooks were called
            assert mock_hook.call_count == 2
            mock_hook.assert_any_call(
                str(repo_dir),
                'pre_gen_project',
                Path(tmpdir) / "hook_test",
                context,
                True
            )
    
    # Test 7: Undefined variable error
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        template_file = repo_dir / "{{undefined_var}}.txt"
        template_file.write_text("test")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"project_name": "test"}))
        
        context = generate_context(str(context_file))
        
        try:
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=tmpdir
            )
            assert False, "Should have raised UndefinedVariableInTemplate"
        except UndefinedVariableInTemplate:
            pass
    
    # Test 8: Keep project on failure
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create template with undefined variable
        template_file = repo_dir / "{{undefined}}.txt"
        template_file.write_text("test")
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"project_name": "test"}))
        
        context = generate_context(str(context_file))
        
        try:
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=tmpdir,
                keep_project_on_failure=True
            )
            assert False, "Should have raised exception"
        except UndefinedVariableInTemplate:
            # Project directory should still exist
            project_dir = Path(tmpdir) / "test"
            assert project_dir.exists()


# LLM-generated content at query #9
#--------------------------

```python
def test_generate_files():
    import tempfile
    import shutil
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import json

    # Test 1: Basic file generation with template rendering
    def test_basic_generation():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create template structure
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            # Create cookiecutter.json
            context_data = {
                "project_name": "TestProject",
                "version": "1.0.0"
            }
            (repo_dir / "cookiecutter.json").write_text(json.dumps(context_data))
            
            # Create a template file
            template_file = repo_dir / "{{cookiecutter.project_name}}.txt"
            template_file.write_text("Version: {{cookiecutter.version}}")
            
            # Generate files
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {
                "cookiecutter": context_data
            }
            
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False
            )
            
            # Verify output
            expected_file = output_dir / "TestProject" / "TestProject.txt"
            assert expected_file.exists()
            assert expected_file.read_text() == "Version: 1.0.0"
            assert result == str(expected_file.parent)

    # Test 2: Overwrite existing directory
    def test_overwrite_if_exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_data = {"project_name": "TestProject"}
            (repo_dir / "cookiecutter.json").write_text(json.dumps(context_data))
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("Content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # Create existing project directory
            existing_dir = output_dir / "TestProject"
            existing_dir.mkdir()
            existing_file = existing_dir / "old.txt"
            existing_file.write_text("Old content")
            
            context = {"cookiecutter": context_data}
            
            # Should raise without overwrite_if_exists
            try:
                generate_files(
                    repo_dir=str(repo_dir),
                    context=context,
                    output_dir=str(output_dir),
                    overwrite_if_exists=False
                )
                assert False, "Should have raised OutputDirExistsException"
            except OutputDirExistsException:
                pass
            
            # Should succeed with overwrite_if_exists
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=True
            )
            
            expected_file = output_dir / "TestProject" / "test.txt"
            assert expected_file.exists()
            assert not existing_file.exists()

    # Test 3: Skip if file exists
    def test_skip_if_file_exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_data = {"project_name": "TestProject"}
            (repo_dir / "cookiecutter.json").write_text(json.dumps(context_data))
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("New content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # Create existing file in project directory
            project_dir = output_dir / "TestProject"
            project_dir.mkdir()
            existing_file = project_dir / "test.txt"
            existing_file.write_text("Existing content")
            
            context = {"cookiecutter": context_data}
            
            # With skip_if_file_exists=True, existing file should not be overwritten
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=True,
                skip_if_file_exists=True
            )
            
            assert existing_file.read_text() == "Existing content"

    # Test 4: Copy without render
    def test_copy_without_render():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_data = {
                "project_name": "TestProject",
                "_copy_without_render": ["static/*", "config.json"]
            }
            (repo_dir / "cookiecutter.json").write_text(json.dumps(context_data))
            
            # Create files that should be copied without rendering
            static_dir = repo_dir / "static"
            static_dir.mkdir()
            (static_dir / "image.png").write_bytes(b"binary data")
            
            config_file = repo_dir / "config.json"
            config_file.write_text('{"key": "{{cookiecutter.project_name}}"}')
            
            # Create template file that should be rendered
            template_file = repo_dir / "README.md"
            template_file.write_text("# {{cookiecutter.project_name}}")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {"cookiecutter": context_data}
            
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir)
            )
            
            # Verify copied files (not rendered)
            copied_config = output_dir / "TestProject" / "config.json"
            assert copied_config.exists()
            assert copied_config.read_text() == '{"key": "{{cookiecutter.project_name}}"}'
            
            # Verify rendered file
            readme = output_dir / "TestProject" / "README.md"
            assert readme.exists()
            assert readme.read_text() == "# TestProject"

    # Test 5: Binary file handling
    def test_binary_file():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_data = {"project_name": "TestProject"}
            (repo_dir / "cookiecutter.json").write_text(json.dumps(context_data))
            
            # Create binary file
            binary_file = repo_dir / "binary.dat"
            binary_file.write_bytes(b"\x00\x01\x02\x03\x04")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {"cookiecutter": context_data}
            
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir)
            )
            
            generated_binary = output_dir / "TestProject" / "binary.dat"
            assert generated_binary.exists()
            assert generated_binary.read_bytes() == b"\x00\x01\x02\x03\x04"

    # Test 6: Undefined variable error
    def test_undefined_variable():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_data = {"project_name": "TestProject"}
            (repo_dir / "cookiecutter.json").write_text(json.dumps(context_data))
            
            # Template with undefined variable
            template_file = repo_dir / "{{undefined_var}}.txt"
            template_file.write_text("Content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {"cookiecutter": context_data}
            
            try:
                generate_files(
                    repo_dir=str(repo_dir),
                    context=context,
                    output_dir=str(output_dir)
                )
                assert False, "Should have raised UndefinedVariableInTemplate"
            except UndefinedVariableInTemplate:
                pass

    # Test 7: Hooks execution
    def test_hooks_execution():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_data = {"project_name": "TestProject"}
            (repo_dir / "cookiecutter.json").write_text(json.dumps(context_data))
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("Content")
            
            # Create hooks directory and pre-gen hook
            hooks_dir = repo_dir / "hooks"
            hooks_dir.mkdir()
            pre_hook = hooks_dir / "pre_gen_project.py"
            pre_hook.write_text("""
import sys
sys.path.insert(0, '.')
with open('pre_hook_ran.txt', 'w') as f:
    f.write('pre hook executed')
""")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {"cookiecutter": context_data}
            
            with patch('cookiecutter.hooks.run_hook_from_repo_dir') as mock_run_hook:
                generate_files(
                    repo_dir=str(repo_dir),
                    context=context,
                    output_dir=str(output_dir),
                    accept_hooks=True
                )
                
                # Verify hooks were called
                assert mock_run_hook.call_count == 2
                assert mock_run_hook.call_args_list[0][0][1] == 'pre_gen_project'
                assert mock_run_hook.call_args_list[1][0][1] == 'post_gen_project'

    # Test 8:


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_files():
    import tempfile
    import shutil
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import json

    # Test 1: Basic file generation with template rendering
    def test_basic_generation():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create template structure
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            # Create cookiecutter.json
            context_data = {
                "project_name": "TestProject",
                "version": "1.0.0"
            }
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump(context_data, f)
            
            # Create a template file
            template_file = repo_dir / "{{cookiecutter.project_name}}.txt"
            template_file.write_text("Version: {{cookiecutter.version}}")
            
            # Create output directory
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # Generate files
            context = {
                "cookiecutter": {
                    "project_name": "TestProject",
                    "version": "1.0.0",
                    "_new_lines": "\n"
                }
            }
            
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False
            )
            
            # Verify the generated file
            expected_file = output_dir / "TestProject" / "TestProject.txt"
            assert expected_file.exists()
            assert expected_file.read_text() == "Version: 1.0.0"
            assert result == str(expected_file.parent)

    # Test 2: Overwrite existing directory
    def test_overwrite_if_exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump({"project_name": "Test"}, f)
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # First generation
            context = {"cookiecutter": {"project_name": "Test", "_new_lines": "\n"}}
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False
            )
            
            # Modify the template
            template_file.write_text("new content")
            
            # Try to generate again without overwrite - should raise exception
            try:
                generate_files(
                    repo_dir=str(repo_dir),
                    context=context,
                    output_dir=str(output_dir),
                    overwrite_if_exists=False
                )
                assert False, "Should have raised OutputDirExistsException"
            except OutputDirExistsException:
                pass
            
            # Generate with overwrite - should succeed
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=True
            )

    # Test 3: Skip if file exists
    def test_skip_if_file_exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump({"project_name": "Test"}, f)
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("original")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            # First generation
            context = {"cookiecutter": {"project_name": "Test", "_new_lines": "\n"}}
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False
            )
            
            # Modify the generated file
            generated_file = output_dir / "Test" / "test.txt"
            generated_file.write_text("modified")
            
            # Generate again with skip_if_file_exists=True
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=True,
                skip_if_file_exists=True
            )
            
            # File should still have modified content
            assert generated_file.read_text() == "modified"

    # Test 4: Copy without render
    def test_copy_without_render():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            context_data = {
                "project_name": "Test",
                "_copy_without_render": ["static/*", "config.json"]
            }
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump(context_data, f)
            
            # Create files that should be copied without rendering
            static_dir = repo_dir / "static"
            static_dir.mkdir()
            (static_dir / "image.png").write_bytes(b"binary data")
            
            config_file = repo_dir / "config.json"
            config_file.write_text('{"key": "{{cookiecutter.project_name}}"}')
            
            # Create a template file that should be rendered
            template_file = repo_dir / "README.md"
            template_file.write_text("# {{cookiecutter.project_name}}")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {
                "cookiecutter": {
                    "project_name": "Test",
                    "_copy_without_render": ["static/*", "config.json"],
                    "_new_lines": "\n"
                }
            }
            
            result = generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False
            )
            
            # Verify copied files maintain original content
            copied_config = output_dir / "Test" / "config.json"
            assert copied_config.exists()
            assert copied_config.read_text() == '{"key": "{{cookiecutter.project_name}}"}'
            
            # Verify rendered file
            readme = output_dir / "Test" / "README.md"
            assert readme.read_text() == "# Test"

    # Test 5: Binary file handling
    def test_binary_file():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump({"project_name": "Test"}, f)
            
            # Create a binary file
            binary_file = repo_dir / "binary.bin"
            binary_file.write_bytes(b"\x00\x01\x02\x03\x04")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {"cookiecutter": {"project_name": "Test", "_new_lines": "\n"}}
            
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False
            )
            
            generated_binary = output_dir / "Test" / "binary.bin"
            assert generated_binary.exists()
            assert generated_binary.read_bytes() == b"\x00\x01\x02\x03\x04"

    # Test 6: Hooks execution
    def test_hooks_execution():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump({"project_name": "Test"}, f)
            
            # Create hooks directory and pre-gen hook
            hooks_dir = repo_dir / "hooks"
            hooks_dir.mkdir()
            pre_hook = hooks_dir / "pre_gen_project.py"
            pre_hook.write_text("""
import os
with open(os.path.join('{{cookiecutter.project_name}}', 'pre_hook.txt'), 'w') as f:
    f.write('pre-hook executed')
""")
            
            template_file = repo_dir / "test.txt"
            template_file.write_text("content")
            
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            
            context = {"cookiecutter": {"project_name": "Test", "_new_lines": "\n"}}
            
            generate_files(
                repo_dir=str(repo_dir),
                context=context,
                output_dir=str(output_dir),
                overwrite_if_exists=False,
                accept_hooks=True
            )
            
            # Verify hook was executed
            hook_file = output_dir / "Test" / "pre_hook.txt"
            assert hook_file.exists()
            assert hook_file.read_text() == "pre-hook executed"

    # Test 7: Undefined variable error
    def test_undefined_variable():
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "template"
            repo_dir.mkdir()
            
            with open(repo_dir / "cookiecutter.json", "w") as f:
                json.dump({"project_name": "Test"}, f)
            
            # Create template with undefined


# LLM-generated content at query #11
#--------------------------

```python
def test_generate_context():
    import json
    import tempfile
    import os
    from collections import OrderedDict
    
    # Test 1: Normal context generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert 'cookiecutter' in context
        assert context['cookiecutter']['project_name'] == 'Test'
        assert context['cookiecutter']['version'] == '1.0'
    finally:
        os.unlink(context_file)
    
    # Test 2: With default_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0"}, f)
        context_file = f.name
    
    try:
        default_context = {"version": "2.0"}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['version'] == "2.0"
        assert context['cookiecutter']['project_name'] == "Test"
    finally:
        os.unlink(context_file)
    
    # Test 3: With extra_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0"}, f)
        context_file = f.name
    
    try:
        extra_context = {"version": "3.0"}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['version'] == "3.0"
    finally:
        os.unlink(context_file)
    
    # Test 4: With both default_context and extra_context (extra should override default)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0"}, f)
        context_file = f.name
    
    try:
        default_context = {"version": "2.0"}
        extra_context = {"version": "3.0"}
        context = generate_context(
            context_file, 
            default_context=default_context, 
            extra_context=extra_context
        )
        assert context['cookiecutter']['version'] == "3.0"
    finally:
        os.unlink(context_file)
    
    # Test 5: Invalid JSON should raise ContextDecodingException
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": json}')
        context_file = f.name
    
    try:
        import pytest
        from cookiecutter.exceptions import ContextDecodingException
        with pytest.raises(ContextDecodingException):
            generate_context(context_file)
    finally:
        os.unlink(context_file)
    
    # Test 6: Complex nested structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "project": {
                "name": "Test",
                "settings": {"debug": True}
            },
            "list": ["a", "b", "c"]
        }, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert context['cookiecutter']['project']['name'] == 'Test'
        assert context['cookiecutter']['project']['settings']['debug'] == True
        assert context['cookiecutter']['list'] == ['a', 'b', 'c']
    finally:
        os.unlink(context_file)
    
    # Test 7: File stem extraction
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"test": "value"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        # The file stem (without extension) should be a key in the context
        file_stem = os.path.splitext(os.path.basename(context_file))[0]
        assert file_stem in context
    finally:
        os.unlink(context_file)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_generate_context():
    import json
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.generate import generate_context, ContextDecodingException
    
    # Test 1: Normal context generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert 'cookiecutter' in context
        assert context['cookiecutter']['project_name'] == "Test Project"
        assert context['cookiecutter']['version'] == "1.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 2: Context generation with default_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        default_context = {"project_name": "Default Project"}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['project_name'] == "Default Project"
        assert context['cookiecutter']['version'] == "1.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 3: Context generation with extra_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        extra_context = {"version": "2.0.0"}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project_name'] == "Test Project"
        assert context['cookiecutter']['version'] == "2.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 4: Context generation with both default and extra context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        default_context = {"project_name": "Default Project"}
        extra_context = {"version": "3.0.0"}
        context = generate_context(
            context_file, 
            default_context=default_context, 
            extra_context=extra_context
        )
        assert context['cookiecutter']['project_name'] == "Default Project"
        assert context['cookiecutter']['version'] == "3.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 5: Invalid JSON file should raise ContextDecodingException
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": json}')
        context_file = f.name
    
    try:
        raised_exception = False
        try:
            generate_context(context_file)
        except ContextDecodingException:
            raised_exception = True
        assert raised_exception
    finally:
        os.unlink(context_file)
    
    # Test 6: Non-existent file should raise FileNotFoundError
    raised_exception = False
    try:
        generate_context("non_existent_file.json")
    except FileNotFoundError:
        raised_exception = True
    assert raised_exception
    
    # Test 7: Context with nested structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "project": {
                "name": "Test",
                "author": "Developer"
            },
            "settings": ["option1", "option2"]
        }, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert context['cookiecutter']['project']['name'] == "Test"
        assert context['cookiecutter']['project']['author'] == "Developer"
        assert context['cookiecutter']['settings'] == ["option1", "option2"]
    finally:
        os.unlink(context_file)
    
    # Test 8: Empty JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert context['cookiecutter'] == {}
    finally:
        os.unlink(context_file)


# LLM-generated content at query #2
#--------------------------

```python
def test_render_and_create_dir():
    import tempfile
    import shutil
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.exceptions import EmptyDirNameException, OutputDirExistsException

    # Create a temporary directory for testing
    tmpdir = tempfile.mkdtemp()
    output_dir = Path(tmpdir)

    # Create a simple Jinja2 environment
    env = Environment()

    # Test 1: Normal directory creation
    context = {"project_name": "myproject"}
    dirname = "{{ project_name }}"
    result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
    expected_dir = output_dir / "myproject"
    assert result_dir == expected_dir
    assert created is True
    assert expected_dir.exists()

    # Test 2: Empty directory name should raise exception
    try:
        render_and_create_dir("", context, output_dir, env)
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException:
        pass

    # Test 3: Directory already exists without overwrite should raise exception
    try:
        render_and_create_dir("myproject", context, output_dir, env)
        assert False, "Should have raised OutputDirExistsException"
    except OutputDirExistsException:
        pass

    # Test 4: Directory already exists with overwrite should succeed
    result_dir, created = render_and_create_dir(
        "myproject", context, output_dir, env, overwrite_if_exists=True
    )
    assert result_dir == expected_dir
    assert created is False  # Directory already existed
    assert expected_dir.exists()

    # Test 5: Complex template rendering
    context = {"user": "john", "project": "test"}
    dirname = "projects/{{ user }}/{{ project }}"
    result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
    expected_dir = output_dir / "projects/john/test"
    assert result_dir == expected_dir
    assert created is True
    assert expected_dir.exists()

    # Test 6: Nested directory creation
    context = {"level1": "a", "level2": "b", "level3": "c"}
    dirname = "{{ level1 }}/{{ level2 }}/{{ level3 }}"
    result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
    expected_dir = output_dir / "a/b/c"
    assert result_dir == expected_dir
    assert created is True
    assert expected_dir.exists()

    # Test 7: Directory name with special characters
    context = {"name": "test-project_123"}
    dirname = "{{ name }}"
    result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
    expected_dir = output_dir / "test-project_123"
    assert result_dir == expected_dir
    assert created is True
    assert expected_dir.exists()

    # Clean up
    shutil.rmtree(tmpdir)


# LLM-generated content at query #3
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test 1: Simple overwrite of existing variable
    context = {"name": "original", "version": "1.0"}
    overwrite = {"name": "new_name"}
    apply_overwrites_to_context(context, overwrite)
    assert context["name"] == "new_name"
    assert context["version"] == "1.0"

    # Test 2: New variable on first level should be ignored
    context = {"existing": "value"}
    overwrite = {"new_var": "new_value"}
    apply_overwrites_to_context(context, overwrite)
    assert "new_var" not in context
    assert context["existing"] == "value"

    # Test 3: Multichoice variable - valid subset
    context = {"choices": ["a", "b", "c", "d"]}
    overwrite = {"choices": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite)
    assert context["choices"] == ["b", "c"]

    # Test 4: Multichoice variable - invalid subset should raise ValueError
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": ["b", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provided for multi-choice variable" in str(e)

    # Test 5: Choice variable - valid choice
    context = {"choice": ["default", "option1", "option2"]}
    overwrite = {"choice": "option2"}
    apply_overwrites_to_context(context, overwrite)
    assert context["choice"] == ["option2", "default", "option1"]

    # Test 6: Choice variable - invalid choice should raise ValueError
    context = {"choice": ["default", "option1"]}
    overwrite = {"choice": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provided for choice variable" in str(e)

    # Test 7: Dictionary variable - partial overwrite
    context = {"config": {"key1": "value1", "key2": "value2"}}
    overwrite = {"config": {"key2": "new_value2", "key3": "value3"}}
    apply_overwrites_to_context(context, overwrite)
    assert context["config"]["key1"] == "value1"
    assert context["config"]["key2"] == "new_value2"
    assert context["config"]["key3"] == "value3"

    # Test 8: Boolean variable - string "yes" conversion
    context = {"flag": True}
    overwrite = {"flag": "no"}
    apply_overwrites_to_context(context, overwrite)
    assert context["flag"] is False

    # Test 9: Boolean variable - string "YES" conversion
    context = {"flag": False}
    overwrite = {"flag": "YES"}
    apply_overwrites_to_context(context, overwrite)
    assert context["flag"] is True

    # Test 10: Boolean variable - invalid string should raise ValueError
    context = {"flag": True}
    overwrite = {"flag": "invalid"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)

    # Test 11: Nested dictionary with in_dictionary_variable=True
    context = {"nested": {"inner": "value"}}
    overwrite = {"new_key": "new_value"}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context["new_key"] == "new_value"
    assert context["nested"]["inner"] == "value"

    # Test 12: List overwrite within dictionary variable
    context = {"data": {"items": ["a", "b", "c"]}}
    overwrite = {"data": {"items": ["x", "y"]}}
    apply_overwrites_to_context(context, overwrite)
    assert context["data"]["items"] == ["x", "y"]

    # Test 13: Mixed types - simple overwrite
    context = {"number": 42, "text": "hello"}
    overwrite = {"number": 100, "text": "world"}
    apply_overwrites_to_context(context, overwrite)
    assert context["number"] == 100
    assert context["text"] == "world"

    # Test 14: Empty overwrite context
    context = {"key": "value"}
    overwrite = {}
    apply_overwrites_to_context(context, overwrite)
    assert context["key"] == "value"

    # Test 15: Overwrite with None value
    context = {"key": "value"}
    overwrite = {"key": None}
    apply_overwrites_to_context(context, overwrite)
    assert context["key"] is None


# LLM-generated content at query #4
#--------------------------

```python
def test_is_copy_only_path():
    # Test case 1: Path matches pattern in _copy_without_render
    context = {
        'cookiecutter': {
            '_copy_without_render': ['*.txt', 'docs/*', 'images/**']
        }
    }
    assert is_copy_only_path('readme.txt', context) == True
    assert is_copy_only_path('docs/index.md', context) == True
    assert is_copy_only_path('images/photo.jpg', context) == True
    assert is_copy_only_path('images/subfolder/photo.png', context) == True
    
    # Test case 2: Path does not match any pattern
    assert is_copy_only_path('src/main.py', context) == False
    assert is_copy_only_path('config.yaml', context) == False
    
    # Test case 3: _copy_without_render key is missing
    context_missing_key = {'cookiecutter': {}}
    assert is_copy_only_path('any/path', context_missing_key) == False
    
    # Test case 4: Empty _copy_without_render list
    context_empty_list = {
        'cookiecutter': {
            '_copy_without_render': []
        }
    }
    assert is_copy_only_path('any/path', context_empty_list) == False
    
    # Test case 5: Exact match pattern
    context_exact = {
        'cookiecutter': {
            '_copy_without_render': ['exact_file.md']
        }
    }
    assert is_copy_only_path('exact_file.md', context_exact) == True
    assert is_copy_only_path('other_file.md', context_exact) == False
    
    # Test case 6: Pattern with wildcards
    context_wildcards = {
        'cookiecutter': {
            '_copy_without_render': ['data/*.csv', 'logs/*.log']
        }
    }
    assert is_copy_only_path('data/2023.csv', context_wildcards) == True
    assert is_copy_only_path('logs/app.log', context_wildcards) == True
    assert is_copy_only_path('data/backup/old.csv', context_wildcards) == False
    
    # Test case 7: Nested patterns
    context_nested = {
        'cookiecutter': {
            '_copy_without_render': ['assets/**/*.png', 'templates/*.html']
        }
    }
    assert is_copy_only_path('assets/images/icon.png', context_nested) == True
    assert is_copy_only_path('assets/textures/wood/diffuse.png', context_nested) == True
    assert is_copy_only_path('templates/index.html', context_nested) == True
    assert is_copy_only_path('assets/readme.txt', context_nested) == False


# LLM-generated content at query #5
#--------------------------

```python
def test_generate_context():
    import json
    import tempfile
    import os
    from cookiecutter.exceptions import ContextDecodingException
    
    # Test 1: Normal context generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert 'cookiecutter' in context
        assert context['cookiecutter']['project_name'] == "Test Project"
        assert context['cookiecutter']['version'] == "1.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 2: With default_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": ["1.0.0", "2.0.0"]}, f)
        context_file = f.name
    
    try:
        default_context = {"version": "2.0.0"}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['version'] == ["2.0.0", "1.0.0"]
    finally:
        os.unlink(context_file)
    
    # Test 3: With extra_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "open_source": True}, f)
        context_file = f.name
    
    try:
        extra_context = {"project_name": "Overridden"}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project_name'] == "Overridden"
    finally:
        os.unlink(context_file)
    
    # Test 4: Invalid JSON should raise ContextDecodingException
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{invalid json")
        context_file = f.name
    
    try:
        try:
            generate_context(context_file)
            assert False, "Should have raised ContextDecodingException"
        except ContextDecodingException:
            pass
    finally:
        os.unlink(context_file)
    
    # Test 5: Complex nested structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "project": {
                "name": "Test",
                "settings": {"debug": True}
            },
            "features": ["auth", "api"]
        }, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert context['cookiecutter']['project']['name'] == "Test"
        assert context['cookiecutter']['project']['settings']['debug'] is True
        assert context['cookiecutter']['features'] == ["auth", "api"]
    finally:
        os.unlink(context_file)
    
    # Test 6: With both default_context and extra_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "name": "Default Name",
            "version": ["1.0", "2.0"],
            "settings": {"debug": False}
        }, f)
        context_file = f.name
    
    try:
        default_context = {"version": "2.0"}
        extra_context = {"name": "Final Name", "settings": {"debug": True}}
        context = generate_context(
            context_file, 
            default_context=default_context, 
            extra_context=extra_context
        )
        assert context['cookiecutter']['name'] == "Final Name"
        assert context['cookiecutter']['version'] == ["2.0", "1.0"]
        assert context['cookiecutter']['settings']['debug'] is True
    finally:
        os.unlink(context_file)


# LLM-generated content at query #6
#--------------------------

```python
def test_generate_context():
    import json
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.exceptions import ContextDecodingException
    
    # Test 1: Normal context generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert 'cookiecutter' in context
        assert context['cookiecutter']['project_name'] == "Test Project"
        assert context['cookiecutter']['version'] == "1.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 2: With default_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        default_context = {"project_name": "Default Project"}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['project_name'] == "Default Project"
    finally:
        os.unlink(context_file)
    
    # Test 3: With extra_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        extra_context = {"project_name": "Extra Project"}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project_name'] == "Extra Project"
    finally:
        os.unlink(context_file)
    
    # Test 4: Invalid JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": json}')
        context_file = f.name
    
    try:
        try:
            generate_context(context_file)
            assert False, "Should have raised ContextDecodingException"
        except ContextDecodingException:
            pass
    finally:
        os.unlink(context_file)
    
    # Test 5: Complex nested structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "project": {
                "name": "Test",
                "settings": {"debug": True}
            },
            "list": ["a", "b", "c"]
        }, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert context['cookiecutter']['project']['name'] == "Test"
        assert context['cookiecutter']['project']['settings']['debug'] is True
        assert context['cookiecutter']['list'] == ["a", "b", "c"]
    finally:
        os.unlink(context_file)
    
    # Test 6: Default and extra context together
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"name": "Original", "version": "1.0"}, f)
        context_file = f.name
    
    try:
        default_context = {"name": "Default"}
        extra_context = {"version": "2.0"}
        context = generate_context(
            context_file, 
            default_context=default_context, 
            extra_context=extra_context
        )
        assert context['cookiecutter']['name'] == "Default"
        assert context['cookiecutter']['version'] == "2.0"
    finally:
        os.unlink(context_file)


# LLM-generated content at query #7
#--------------------------

```python
def test_generate_context():
    import json
    import tempfile
    import os
    from collections import OrderedDict
    from cookiecutter.exceptions import ContextDecodingException
    
    # Test 1: Basic context generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert 'cookiecutter' in context
        assert context['cookiecutter']['project_name'] == "Test Project"
        assert context['cookiecutter']['version'] == "1.0"
    finally:
        os.unlink(context_file)
    
    # Test 2: With default_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0"}, f)
        context_file = f.name
    
    try:
        default_context = {"project_name": "Default Project"}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['project_name'] == "Default Project"
        assert context['cookiecutter']['version'] == "1.0"
    finally:
        os.unlink(context_file)
    
    # Test 3: With extra_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0"}, f)
        context_file = f.name
    
    try:
        extra_context = {"project_name": "Extra Project"}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project_name'] == "Extra Project"
        assert context['cookiecutter']['version'] == "1.0"
    finally:
        os.unlink(context_file)
    
    # Test 4: With both default_context and extra_context (extra should override)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Original", "version": "1.0"}, f)
        context_file = f.name
    
    try:
        default_context = {"project_name": "Default Project"}
        extra_context = {"project_name": "Extra Project"}
        context = generate_context(
            context_file, 
            default_context=default_context, 
            extra_context=extra_context
        )
        assert context['cookiecutter']['project_name'] == "Extra Project"
    finally:
        os.unlink(context_file)
    
    # Test 5: Invalid JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{invalid json")
        context_file = f.name
    
    try:
        try:
            generate_context(context_file)
            assert False, "Should have raised ContextDecodingException"
        except ContextDecodingException:
            pass
    finally:
        os.unlink(context_file)
    
    # Test 6: Complex nested structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "project": {
                "name": "Test",
                "settings": {"debug": True}
            },
            "list": ["a", "b", "c"]
        }, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert context['cookiecutter']['project']['name'] == "Test"
        assert context['cookiecutter']['project']['settings']['debug'] is True
        assert context['cookiecutter']['list'] == ["a", "b", "c"]
    finally:
        os.unlink(context_file)
    
    # Test 7: File with different name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"test": "value"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        file_stem = os.path.split(context_file)[1].split('.')[0]
        assert file_stem in context
        assert context[file_stem]['test'] == "value"
    finally:
        os.unlink(context_file)


# LLM-generated content at query #8
#--------------------------

```python
def test_render_and_create_dir():
    import tempfile
    import shutil
    from pathlib import Path
    from jinja2 import Environment
    
    # Create a temporary directory for testing
    tmpdir = tempfile.mkdtemp()
    
    try:
        # Test 1: Normal directory creation with rendered name
        context = {'project_name': 'MyProject'}
        env = Environment()
        output_dir = Path(tmpdir)
        
        result_dir, created = render_and_create_dir(
            dirname='{{ project_name }}_app',
            context=context,
            output_dir=output_dir,
            environment=env,
            overwrite_if_exists=False
        )
        
        expected_dir = output_dir / 'MyProject_app'
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
        assert expected_dir.is_dir()
        
        # Test 2: Directory already exists without overwrite
        # Should raise OutputDirExistsException
        try:
            render_and_create_dir(
                dirname='{{ project_name }}_app',
                context=context,
                output_dir=output_dir,
                environment=env,
                overwrite_if_exists=False
            )
            assert False, "Should have raised OutputDirExistsException"
        except OutputDirExistsException as e:
            assert 'already exists' in str(e)
        
        # Test 3: Directory already exists with overwrite
        result_dir, created = render_and_create_dir(
            dirname='{{ project_name }}_app',
            context=context,
            output_dir=output_dir,
            environment=env,
            overwrite_if_exists=True
        )
        
        assert result_dir == expected_dir
        assert created is False  # Directory already existed
        assert expected_dir.exists()
        
        # Test 4: Empty directory name
        try:
            render_and_create_dir(
                dirname='',
                context=context,
                output_dir=output_dir,
                environment=env,
                overwrite_if_exists=False
            )
            assert False, "Should have raised EmptyDirNameException"
        except EmptyDirNameException as e:
            assert 'directory name is empty' in str(e)
        
        # Test 5: Complex template rendering
        context = {
            'user': 'john',
            'version': '1.0',
            'type': 'backend'
        }
        
        result_dir, created = render_and_create_dir(
            dirname='{{ user }}-{{ type }}-v{{ version }}',
            context=context,
            output_dir=output_dir,
            environment=env,
            overwrite_if_exists=False
        )
        
        expected_dir = output_dir / 'john-backend-v1.0'
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
        
        # Test 6: Nested directory path
        result_dir, created = render_and_create_dir(
            dirname='projects/{{ user }}/src',
            context=context,
            output_dir=output_dir,
            environment=env,
            overwrite_if_exists=False
        )
        
        expected_dir = output_dir / 'projects' / 'john' / 'src'
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
        
        # Test 7: Directory name with special characters
        context = {'name': 'test-project_123'}
        result_dir, created = render_and_create_dir(
            dirname='{{ name }}',
            context=context,
            output_dir=output_dir,
            environment=env,
            overwrite_if_exists=False
        )
        
        expected_dir = output_dir / 'test-project_123'
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
        
        # Test 8: Path object as output_dir
        result_dir, created = render_and_create_dir(
            dirname='path_object_test',
            context={},
            output_dir=Path(tmpdir),
            environment=env,
            overwrite_if_exists=False
        )
        
        expected_dir = Path(tmpdir) / 'path_object_test'
        assert result_dir == expected_dir
        assert created is True
        
        # Test 9: String as output_dir
        result_dir, created = render_and_create_dir(
            dirname='string_output_test',
            context={},
            output_dir=tmpdir,
            environment=env,
            overwrite_if_exists=False
        )
        
        expected_dir = Path(tmpdir) / 'string_output_test'
        assert result_dir == expected_dir
        assert created is True
        
    finally:
        # Clean up
        shutil.rmtree(tmpdir)


# LLM-generated content at query #9
#--------------------------

```python
def test_render_and_create_dir():
    import tempfile
    import shutil
    from pathlib import Path
    from jinja2 import Environment
    
    # Setup test environment
    test_dir = tempfile.mkdtemp()
    output_dir = Path(test_dir) / "output"
    output_dir.mkdir(parents=True)
    
    # Create a simple Jinja2 environment
    env = Environment()
    
    # Test 1: Normal directory creation with rendered name
    context = {"project_name": "MyProject"}
    dirname = "{{ project_name }}_dir"
    
    result_path, created = render_and_create_dir(dirname, context, output_dir, env)
    
    assert result_path == output_dir / "MyProject_dir"
    assert created is True
    assert result_path.exists()
    assert result_path.is_dir()
    
    # Test 2: Directory already exists without overwrite
    # Create the directory first
    existing_dir = output_dir / "ExistingDir"
    existing_dir.mkdir()
    
    dirname = "ExistingDir"
    try:
        render_and_create_dir(dirname, {}, output_dir, env, overwrite_if_exists=False)
        assert False, "Should have raised OutputDirExistsException"
    except OutputDirExistsException as e:
        assert "already exists" in str(e)
    
    # Test 3: Directory already exists with overwrite
    result_path, created = render_and_create_dir(
        dirname, {}, output_dir, env, overwrite_if_exists=True
    )
    
    assert result_path == existing_dir
    assert created is False  # Directory already existed
    assert existing_dir.exists()  # Should still exist
    
    # Test 4: Empty directory name
    try:
        render_and_create_dir("", {}, output_dir, env)
        assert False, "Should have raised EmptyDirNameException"
    except EmptyDirNameException as e:
        assert "directory name is empty" in str(e)
    
    # Test 5: Complex template rendering
    context = {"user": "testuser", "version": "1.0"}
    dirname = "app_{{ user }}_{{ version }}"
    
    result_path, created = render_and_create_dir(dirname, context, output_dir, env)
    
    assert result_path == output_dir / "app_testuser_1.0"
    assert created is True
    assert result_path.exists()
    
    # Test 6: Nested path in template
    dirname = "src/{{ project_name }}/tests"
    context = {"project_name": "MyApp"}
    
    result_path, created = render_and_create_dir(dirname, context, output_dir, env)
    
    expected_path = output_dir / "src" / "MyApp" / "tests"
    assert result_path == expected_path
    assert created is True
    assert expected_path.exists()
    
    # Test 7: Template with special characters
    context = {"name": "test-project"}
    dirname = "{{ name }}_folder"
    
    result_path, created = render_and_create_dir(dirname, context, output_dir, env)
    
    assert result_path == output_dir / "test-project_folder"
    assert created is True
    
    # Cleanup
    shutil.rmtree(test_dir)


# LLM-generated content at query #10
#--------------------------

```python
def test_generate_context():
    import json
    import tempfile
    import os
    from pathlib import Path
    from cookiecutter.generate import generate_context, ContextDecodingException
    
    # Test 1: Normal context generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert 'cookiecutter' in context
        assert context['cookiecutter']['project_name'] == "Test Project"
        assert context['cookiecutter']['version'] == "1.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 2: With default context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        default_context = {"project_name": "Default Project"}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['project_name'] == "Default Project"
        assert context['cookiecutter']['version'] == "1.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 3: With extra context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        extra_context = {"project_name": "Overridden Project"}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project_name'] == "Overridden Project"
        assert context['cookiecutter']['version'] == "1.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 4: Invalid JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": json}')
        context_file = f.name
    
    try:
        try:
            generate_context(context_file)
            assert False, "Should have raised ContextDecodingException"
        except ContextDecodingException:
            pass
    finally:
        os.unlink(context_file)
    
    # Test 5: Complex nested structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "project": {
                "name": "Test",
                "settings": {"debug": True}
            },
            "list": ["a", "b", "c"]
        }, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert context['cookiecutter']['project']['name'] == "Test"
        assert context['cookiecutter']['project']['settings']['debug'] is True
        assert context['cookiecutter']['list'] == ["a", "b", "c"]
    finally:
        os.unlink(context_file)
    
    # Test 6: File with different name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"key": "value"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        file_stem = Path(context_file).stem
        assert file_stem in context
    finally:
        os.unlink(context_file)


# LLM-generated content at query #11
#--------------------------

```python
def test_render_and_create_dir():
    import tempfile
    from pathlib import Path
    from jinja2 import Environment
    from cookiecutter.exceptions import EmptyDirNameException, OutputDirExistsException
    
    # Test 1: Empty directory name should raise EmptyDirNameException
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "test_project"}
        output_dir = Path(tmpdir)
        
        try:
            render_and_create_dir("", context, output_dir, env)
            assert False, "Should have raised EmptyDirNameException"
        except EmptyDirNameException as e:
            assert "directory name is empty" in str(e)
    
    # Test 2: Directory name with only whitespace should raise EmptyDirNameException
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "test_project"}
        output_dir = Path(tmpdir)
        
        try:
            render_and_create_dir("   ", context, output_dir, env)
            assert False, "Should have raised EmptyDirNameException"
        except EmptyDirNameException as e:
            assert "directory name is empty" in str(e)
    
    # Test 3: Successful directory creation with rendered name
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "my_project"}
        output_dir = Path(tmpdir)
        
        dirname = "{{ project_name }}_dir"
        result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
        
        expected_dir = output_dir / "my_project_dir"
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
        assert expected_dir.is_dir()
    
    # Test 4: Directory already exists without overwrite should raise OutputDirExistsException
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "existing_project"}
        output_dir = Path(tmpdir)
        
        # Create the directory first
        existing_dir = output_dir / "existing_project_dir"
        existing_dir.mkdir()
        
        dirname = "{{ project_name }}_dir"
        try:
            render_and_create_dir(dirname, context, output_dir, env, overwrite_if_exists=False)
            assert False, "Should have raised OutputDirExistsException"
        except OutputDirExistsException as e:
            assert "already exists" in str(e)
            assert str(existing_dir) in str(e)
    
    # Test 5: Directory already exists with overwrite should succeed
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "overwritten_project"}
        output_dir = Path(tmpdir)
        
        # Create the directory first
        existing_dir = output_dir / "overwritten_project_dir"
        existing_dir.mkdir()
        
        dirname = "{{ project_name }}_dir"
        result_dir, created = render_and_create_dir(
            dirname, context, output_dir, env, overwrite_if_exists=True
        )
        
        assert result_dir == existing_dir
        assert created is False  # Directory already existed
        assert existing_dir.exists()
    
    # Test 6: Complex Jinja2 template in directory name
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {
            "project_name": "complex",
            "version": "1.0",
            "author": "test_author"
        }
        output_dir = Path(tmpdir)
        
        dirname = "{{ project_name }}-v{{ version }}-by-{{ author }}"
        result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
        
        expected_dir = output_dir / "complex-v1.0-by-test_author"
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
    
    # Test 7: Nested directory path creation
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "nested_project"}
        output_dir = Path(tmpdir)
        
        dirname = "projects/{{ project_name }}/src"
        result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
        
        expected_dir = output_dir / "projects" / "nested_project" / "src"
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
        assert (output_dir / "projects").exists()
        assert (output_dir / "projects" / "nested_project").exists()
    
    # Test 8: Directory name with special characters
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"name": "test@project#123"}
        output_dir = Path(tmpdir)
        
        dirname = "{{ name }}"
        result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
        
        expected_dir = output_dir / "test@project#123"
        assert result_dir == expected_dir
        assert created is True
        assert expected_dir.exists()
    
    # Test 9: String output_dir instead of Path
    with tempfile.TemporaryDirectory() as tmpdir:
        env = Environment()
        context = {"project_name": "string_path"}
        output_dir = tmpdir  # String instead of Path
        
        dirname = "{{ project_name }}_test"
        result_dir, created = render_and_create_dir(dirname, context, output_dir, env)
        
        import os
        expected_path = os.path.join(tmpdir, "string_path_test")
        assert str(result_dir) == expected_path
        assert created is True
        assert os.path.exists(expected_path)
        assert os.path.isdir(expected_path)


# LLM-generated content at query #12
#--------------------------

```python
def test_generate_context():
    import json
    import tempfile
    import os
    from collections import OrderedDict
    
    # Test 1: Normal context generation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test Project", "version": "1.0.0"}, f)
        context_file = f.name
    
    try:
        context = generate_context(context_file)
        assert 'cookiecutter' in context
        assert context['cookiecutter']['project_name'] == "Test Project"
        assert context['cookiecutter']['version'] == "1.0.0"
    finally:
        os.unlink(context_file)
    
    # Test 2: With default_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "version": ["1.0.0", "2.0.0"]}, f)
        context_file = f.name
    
    try:
        default_context = {"version": "2.0.0"}
        context = generate_context(context_file, default_context=default_context)
        assert context['cookiecutter']['version'] == ["2.0.0", "1.0.0"]
    finally:
        os.unlink(context_file)
    
    # Test 3: With extra_context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"project_name": "Test", "open_source": True}, f)
        context_file = f.name
    
    try:
        extra_context = {"project_name": "Overridden", "open_source": "no"}
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project_name'] == "Overridden"
        assert context['cookiecutter']['open_source'] is False
    finally:
        os.unlink(context_file)
    
    # Test 4: Invalid JSON raises ContextDecodingException
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{"invalid": json}')
        context_file = f.name
    
    try:
        try:
            generate_context(context_file)
            assert False, "Should have raised ContextDecodingException"
        except ContextDecodingException:
            pass
    finally:
        os.unlink(context_file)
    
    # Test 5: Complex nested structure
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "project": {
                "name": "Test",
                "config": {"debug": True}
            },
            "choices": ["opt1", "opt2", "opt3"]
        }, f)
        context_file = f.name
    
    try:
        extra_context = {
            "project": {"config": {"debug": False}},
            "choices": "opt2"
        }
        context = generate_context(context_file, extra_context=extra_context)
        assert context['cookiecutter']['project']['config']['debug'] is False
        assert context['cookiecutter']['choices'] == ["opt2", "opt1", "opt3"]
    finally:
        os.unlink(context_file)
    
    # Test 6: Warning for invalid default context
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"choice": ["a", "b", "c"]}, f)
        context_file = f.name
    
    try:
        import warnings
        default_context = {"choice": "invalid"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            context = generate_context(context_file, default_context=default_context)
            assert len(w) == 1
            assert "Invalid default received" in str(w[0].message)
    finally:
        os.unlink(context_file)


# LLM-generated content at query #13
#--------------------------

```python
def test_render_and_create_dir():
    import tempfile
    import shutil
    from pathlib import Path
    from jinja2 import Environment
    
    # Setup test environment
    test_dir = tempfile.mkdtemp()
    output_dir = Path(test_dir) / "output"
    output_dir.mkdir()
    
    context = {
        "cookiecutter": {
            "project_name": "TestProject",
            "version": "1.0.0"
        }
    }
    
    env = Environment()
    
    try:
        # Test 1: Normal directory creation with template rendering
        dirname = "{{ cookiecutter.project_name }}_{{ cookiecutter.version }}"
        result_path, created = render_and_create_dir(
            dirname, context, output_dir, env
        )
        
        expected_path = output_dir / "TestProject_1.0.0"
        assert result_path == expected_path
        assert created is True
        assert expected_path.exists()
        assert expected_path.is_dir()
        
        # Test 2: Directory already exists without overwrite
        # Should raise OutputDirExistsException
        try:
            render_and_create_dir(dirname, context, output_dir, env)
            assert False, "Should have raised OutputDirExistsException"
        except OutputDirExistsException as e:
            assert "already exists" in str(e)
        
        # Test 3: Directory already exists with overwrite
        result_path, created = render_and_create_dir(
            dirname, context, output_dir, env, overwrite_if_exists=True
        )
        assert result_path == expected_path
        assert created is False  # Directory already existed
        assert expected_path.exists()
        
        # Test 4: Empty directory name
        try:
            render_and_create_dir("", context, output_dir, env)
            assert False, "Should have raised EmptyDirNameException"
        except EmptyDirNameException as e:
            assert "directory name is empty" in str(e)
        
        # Test 5: Complex template with nested directories
        dirname = "projects/{{ cookiecutter.project_name }}/src"
        result_path, created = render_and_create_dir(
            dirname, context, output_dir, env
        )
        
        expected_path = output_dir / "projects/TestProject/src"
        assert result_path == expected_path
        assert created is True
        assert expected_path.exists()
        assert expected_path.is_dir()
        
        # Test 6: Template with undefined variable (should raise UndefinedError)
        dirname = "{{ cookiecutter.undefined_var }}"
        try:
            render_and_create_dir(dirname, context, output_dir, env)
            assert False, "Should have raised UndefinedError"
        except UndefinedError:
            pass  # Expected
        
        # Test 7: Simple directory name (no template variables)
        simple_dir = output_dir / "simple"
        simple_dir.mkdir()
        
        result_path, created = render_and_create_dir(
            "simple", context, output_dir, env
        )
        assert result_path == simple_dir
        assert created is False  # Already existed
        
        # Test 8: Directory name with special characters in template
        context_with_special = {
            "cookiecutter": {
                "name": "Test@Project#123"
            }
        }
        dirname = "{{ cookiecutter.name }}"
        result_path, created = render_and_create_dir(
            dirname, context_with_special, output_dir, env
        )
        
        expected_path = output_dir / "Test@Project#123"
        assert result_path == expected_path
        assert created is True
        assert expected_path.exists()
        
        # Test 9: Path traversal attempt (should be handled by Path)
        dirname = "../outside_dir"
        result_path, created = render_and_create_dir(
            dirname, context, output_dir, env
        )
        # Should create directory relative to output_dir
        expected_path = output_dir / "../outside_dir"
        assert result_path == expected_path
        
        # Test 10: Multiple levels deep with template
        dirname = "a/b/c/{{ cookiecutter.project_name }}"
        result_path, created = render_and_create_dir(
            dirname, context, output_dir, env
        )
        
        expected_path = output_dir / "a/b/c/TestProject"
        assert result_path == expected_path
        assert created is True
        assert expected_path.exists()
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)


# LLM-generated content at query #14
#--------------------------

```python
def test_generate_file():
    import tempfile
    import os
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        template_dir = Path(tmpdir) / "template"
        template_dir.mkdir()
        project_dir = Path(tmpdir) / "project"
        project_dir.mkdir()
        
        # Create a simple text template file
        template_file = template_dir / "test.txt.j2"
        template_file.write_text("Hello {{ name }}!")
        
        # Create a binary file
        binary_file = template_dir / "binary.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03')
        
        # Set up Jinja2 environment
        env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Test 1: Render text file with context
        context = {"name": "World", "cookiecutter": {"_new_lines": "\n"}}
        generate_file(str(project_dir), "test.txt.j2", context, env)
        
        output_file = project_dir / "test.txt"
        assert output_file.exists()
        assert output_file.read_text() == "Hello World!"
        
        # Test 2: Copy binary file without rendering
        context = {"cookiecutter": {}}
        generate_file(str(project_dir), "binary.bin", context, env)
        
        binary_output = project_dir / "binary.bin"
        assert binary_output.exists()
        assert binary_output.read_bytes() == b'\x00\x01\x02\x03'
        
        # Test 3: Skip if file exists
        existing_file = project_dir / "existing.txt"
        existing_file.write_text("Original content")
        
        # Create template for existing file
        existing_template = template_dir / "existing.txt.j2"
        existing_template.write_text("New content")
        
        generate_file(str(project_dir), "existing.txt.j2", context, env, skip_if_file_exists=True)
        assert existing_file.read_text() == "Original content"
        
        # Test 4: File with empty name (directory)
        empty_name_file = template_dir / "empty_dir" / ".keep"
        empty_name_file.parent.mkdir()
        empty_name_file.write_text("content")
        
        # This should create a directory, not a file
        generate_file(str(project_dir), "empty_dir/.keep", context, env)
        assert (project_dir / "empty_dir" / ".keep").exists()
        
        # Test 5: Template syntax error
        bad_template = template_dir / "bad.txt.j2"
        bad_template.write_text("Hello {{ name }")  # Missing closing braces
        
        try:
            generate_file(str(project_dir), "bad.txt.j2", context, env)
            assert False, "Should have raised TemplateSyntaxError"
        except Exception as e:
            assert "TemplateSyntaxError" in str(type(e).__name__)
        
        # Test 6: Undefined variable in template
        undefined_template = template_dir / "undefined.txt.j2"
        undefined_template.write_text("Hello {{ undefined_var }}!")
        
        try:
            generate_file(str(project_dir), "undefined.txt.j2", context, env)
            assert False, "Should have raised UndefinedError"
        except Exception as e:
            assert "UndefinedError" in str(type(e).__name__)
        
        # Test 7: Custom newline character
        newline_template = template_dir / "newline.txt.j2"
        newline_template.write_text("Line1\nLine2\r\nLine3")
        
        context_with_newline = {"cookiecutter": {"_new_lines": "\r\n"}}
        generate_file(str(project_dir), "newline.txt.j2", context_with_newline, env)
        
        newline_output = project_dir / "newline.txt"
        with open(newline_output, 'rb') as f:
            content = f.read()
            assert b'\r\n' in content
        
        # Test 8: File permissions are preserved
        import stat
        import platform
        
        if platform.system() != "Windows":
            # Make template file executable
            template_file.chmod(template_file.stat().st_mode | stat.S_IEXEC)
            
            generate_file(str(project_dir), "test.txt.j2", context, env)
            assert os.access(output_file, os.X_OK)


# LLM-generated content at query #15
#--------------------------

```python
def test_generate_files():
    import tempfile
    import json
    from pathlib import Path
    from cookiecutter.generate import generate_files
    from cookiecutter.exceptions import (
        OutputDirExistsException,
        UndefinedVariableInTemplate,
    )

    # Test 1: Basic file generation with simple template
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        # Create cookiecutter.json
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({
            "project_name": "Test Project",
            "version": "1.0.0"
        }))
        
        # Create a simple template file
        template_file = repo_dir / "{{ cookiecutter.project_name }}.txt"
        template_file.write_text("Version: {{ cookiecutter.version }}")
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        # Generate files
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=str(output_dir),
            overwrite_if_exists=False
        )
        
        # Verify the generated file
        generated_file = output_dir / "Test Project.txt"
        assert generated_file.exists()
        assert generated_file.read_text() == "Version: 1.0.0"
        assert result == str(generated_file.parent)

    # Test 2: Overwrite existing directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"name": "Test"}))
        
        template_file = repo_dir / "test.txt"
        template_file.write_text("content")
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        # Create existing project directory
        existing_dir = output_dir / "Test"
        existing_dir.mkdir()
        
        # Should raise exception without overwrite_if_exists
        try:
            generate_files(
                repo_dir=str(repo_dir),
                output_dir=str(output_dir),
                overwrite_if_exists=False
            )
            assert False, "Should have raised OutputDirExistsException"
        except OutputDirExistsException:
            pass
        
        # Should succeed with overwrite_if_exists
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=str(output_dir),
            overwrite_if_exists=True
        )
        assert Path(result).exists()

    # Test 3: Skip if file exists
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"name": "Test"}))
        
        template_file = repo_dir / "test.txt"
        template_file.write_text("new content")
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        # Create existing file in output
        existing_file = output_dir / "Test" / "test.txt"
        existing_file.parent.mkdir()
        existing_file.write_text("old content")
        
        # Generate with skip_if_file_exists=True
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=str(output_dir),
            overwrite_if_exists=True,
            skip_if_file_exists=True
        )
        
        # File should still have old content
        assert existing_file.read_text() == "old content"

    # Test 4: Undefined variable in template
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"defined_var": "value"}))
        
        template_file = repo_dir / "{{ undefined_var }}.txt"
        template_file.write_text("content")
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        try:
            generate_files(
                repo_dir=str(repo_dir),
                output_dir=str(output_dir)
            )
            assert False, "Should have raised UndefinedVariableInTemplate"
        except UndefinedVariableInTemplate:
            pass

    # Test 5: Copy without render functionality
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({
            "project_name": "MyProject",
            "_copy_without_render": ["static/*", "config.json"]
        }))
        
        # Create directory structure
        static_dir = repo_dir / "static"
        static_dir.mkdir()
        
        # Create files that should be copied without rendering
        static_file = static_dir / "image.png"
        static_file.write_bytes(b"binary content")
        
        config_file = repo_dir / "config.json"
        config_file.write_text('{"key": "{{ cookiecutter.project_name }}"}')
        
        # Create template file that should be rendered
        template_file = repo_dir / "README.md"
        template_file.write_text("# {{ cookiecutter.project_name }}")
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=str(output_dir)
        )
        
        # Verify rendered file
        readme = Path(result) / "README.md"
        assert readme.exists()
        assert readme.read_text() == "# MyProject"
        
        # Verify copied files (should not have template rendered)
        config_output = Path(result) / "config.json"
        assert config_output.exists()
        assert '{{ cookiecutter.project_name }}' in config_output.read_text()
        
        static_output = Path(result) / "static" / "image.png"
        assert static_output.exists()
        assert static_output.read_bytes() == b"binary content"

    # Test 6: With extra context
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({
            "project_name": "Default",
            "version": "1.0.0"
        }))
        
        template_file = repo_dir / "{{ cookiecutter.project_name }}.txt"
        template_file.write_text("Version: {{ cookiecutter.version }}")
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        # Generate with extra context
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=str(output_dir),
            extra_context={"project_name": "Custom"}
        )
        
        # Should use custom name from extra_context
        generated_file = output_dir / "Custom.txt"
        assert generated_file.exists()
        assert generated_file.read_text() == "Version: 1.0.0"

    # Test 7: Binary file handling
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({"name": "Test"}))
        
        # Create a binary file
        binary_file = repo_dir / "binary.dat"
        binary_file.write_bytes(b"\x00\x01\x02\x03\x04")
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=str(output_dir)
        )
        
        # Binary file should be copied without modification
        output_binary = Path(result) / "binary.dat"
        assert output_binary.exists()
        assert output_binary.read_bytes() == b"\x00\x01\x02\x03\x04"

    # Test 8: Nested directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "template"
        repo_dir.mkdir()
        
        context_file = repo_dir / "cookiecutter.json"
        context_file.write_text(json.dumps({
            "project_name": "NestedTest",
            "module_name": "mymodule"
        }))
        
        # Create nested directory structure
        nested_dir = repo_dir / "src" / "{{ cookiecutter.module_name }}"
        nested_dir.mkdir(parents=True)
        
        init_file = nested_dir / "__init__.py"
        init_file.write_text("# {{ cookiecutter.project_name }}")
        
        main_file = nested_dir / "main.py"
        main_file.write_text("print('{{ cookiecutter.project_name }}')")
        
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        result = generate_files(
            repo_dir=str(repo_dir),
            output_dir=str(output_dir)
        )
        
        # Verify nested structure was created
        output_module = Path(result) / "src" / "mymodule"
        assert output_module.exists()
        assert output_module.is_dir()
        
        output_init = output_module / "__init__.py"
        assert output_init.exists()
       


# LLM-generated content at query #16
#--------------------------

```python
def test_apply_overwrites_to_context():
    # Test 1: Simple overwrite of existing variable
    context = {"name": "old", "version": "1.0"}
    overwrite = {"name": "new"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"name": "new", "version": "1.0"}

    # Test 2: New variable on first level should be ignored
    context = {"name": "old"}
    overwrite = {"new_var": "value"}
    apply_overwrites_to_context(context, overwrite)
    assert context == {"name": "old"}

    # Test 3: Overwrite list variable with valid choice
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": "b"}
    apply_overwrites_to_context(context, overwrite)
    assert context["choices"] == ["b", "a", "c"]

    # Test 4: Overwrite list variable with invalid choice raises ValueError
    context = {"choices": ["a", "b", "c"]}
    overwrite = {"choices": "d"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provided for choice variable" in str(e)

    # Test 5: Overwrite multichoice variable with valid subset
    context = {"multichoice": ["a", "b", "c", "d"]}
    overwrite = {"multichoice": ["b", "c"]}
    apply_overwrites_to_context(context, overwrite)
    assert context["multichoice"] == ["b", "c"]

    # Test 6: Overwrite multichoice variable with invalid subset raises ValueError
    context = {"multichoice": ["a", "b", "c"]}
    overwrite = {"multichoice": ["b", "d"]}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "provided for multi-choice variable" in str(e)

    # Test 7: Partial overwrite of nested dictionary
    context = {"config": {"name": "old", "version": "1.0", "enabled": True}}
    overwrite = {"config": {"name": "new", "extra": "value"}}
    apply_overwrites_to_context(context, overwrite)
    assert context["config"] == {"name": "new", "version": "1.0", "enabled": True, "extra": "value"}

    # Test 8: Overwrite boolean variable with string "yes"
    context = {"enabled": False}
    overwrite = {"enabled": "yes"}
    apply_overwrites_to_context(context, overwrite)
    assert context["enabled"] is True

    # Test 9: Overwrite boolean variable with string "no"
    context = {"enabled": True}
    overwrite = {"enabled": "no"}
    apply_overwrites_to_context(context, overwrite)
    assert context["enabled"] is False

    # Test 10: Overwrite boolean variable with invalid string raises ValueError
    context = {"enabled": True}
    overwrite = {"enabled": "maybe"}
    try:
        apply_overwrites_to_context(context, overwrite)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "could not be converted to a boolean" in str(e)

    # Test 11: Overwrite list variable with list when in_dictionary_variable=True
    context = {"nested": {"items": ["a", "b"]}}
    overwrite = {"nested": {"items": ["c", "d"]}}
    apply_overwrites_to_context(context, overwrite, in_dictionary_variable=True)
    assert context["nested"]["items"] == ["c", "d"]

    # Test 12: Complex nested structure
    context = {
        "project": {
            "name": "test",
            "settings": {"debug": False, "log_level": "info"},
            "features": ["auth", "api"]
        }
    }
    overwrite = {
        "project": {
            "name": "updated",
            "settings": {"debug": "yes", "new_setting": "value"},
            "features": ["api"]
        }
    }
    apply_overwrites_to_context(context, overwrite)
    assert context["project"]["name"] == "updated"
    assert context["project"]["settings"]["debug"] is True
    assert context["project"]["settings"]["log_level"] == "info"
    assert context["project"]["settings"]["new_setting"] == "value"
    assert context["project"]["features"] == ["api", "auth"]


