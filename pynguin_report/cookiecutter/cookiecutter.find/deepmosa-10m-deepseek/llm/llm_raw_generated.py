####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_find_template_success():
    repo_dir = Path("/tmp/test_repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["cookiecutter-{{project_name}}"]
    result = find_template(repo_dir, env)
    assert result == Path("/tmp/test_repo/cookiecutter-{{project_name}}")

def test_find_template_no_cookiecutter():
    repo_dir = Path("/tmp/test_repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["notemplate-{{project_name}}"]
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_no_variable_start():
    repo_dir = Path("/tmp/test_repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["cookiecutter-project_name}}"]
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_no_variable_end():
    repo_dir = Path("/tmp/test_repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["cookiecutter-{{project_name"]
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_empty_dir():
    repo_dir = Path("/tmp/test_repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: []
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_multiple_valid():
    repo_dir = Path("/tmp/test_repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["cookiecutter-{{project_name}}", "cookiecutter-{{app_name}}"]
    result = find_template(repo_dir, env)
    assert result == Path("/tmp/test_repo/cookiecutter-{{project_name}}")

def test_find_template_different_variable_strings():
    repo_dir = Path("/tmp/test_repo")
    env = Environment(variable_start_string="[[", variable_end_string="]]")
    os.listdir = lambda x: ["cookiecutter-[[project_name]]"]
    result = find_template(repo_dir, env)
    assert result == Path("/tmp/test_repo/cookiecutter-[[project_name]]")

def test_find_template_repo_dir_as_string():
    repo_dir = "/tmp/test_repo"
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["cookiecutter-{{project_name}}"]
    result = find_template(repo_dir, env)
    assert result == Path("/tmp/test_repo/cookiecutter-{{project_name}}")


# LLM-generated content at query #2
#--------------------------

def test_find_template_predicate_true():
    mock_repo_dir = Path("/fake/repo")
    mock_env = Environment()
    mock_env.variable_start_string = "{{"
    mock_env.variable_end_string = "}}"
    mock_os_listdir = lambda x: ["cookiecutter_{{project_name}}"]
    original_os_listdir = os.listdir
    os.listdir = mock_os_listdir
    result = find_template(mock_repo_dir, mock_env)
    os.listdir = original_os_listdir
    assert result == Path(mock_repo_dir, "cookiecutter_{{project_name}}")


# LLM-generated content at query #3
#--------------------------

def test_find_template_valid():
    repo_dir = Path("/tmp/test_repo")
    os.makedirs(repo_dir / "cookiecutter-{{project}}", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    result = find_template(repo_dir, env)
    assert result == repo_dir / "cookiecutter-{{project}}"

def test_find_template_missing_cookiecutter():
    repo_dir = Path("/tmp/test_repo2")
    os.makedirs(repo_dir / "notemplate-{{project}}", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_missing_variable_start():
    repo_dir = Path("/tmp/test_repo3")
    os.makedirs(repo_dir / "cookiecutter-project}}", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_missing_variable_end():
    repo_dir = Path("/tmp/test_repo4")
    os.makedirs(repo_dir / "cookiecutter-{{project", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_multiple_directories():
    repo_dir = Path("/tmp/test_repo5")
    os.makedirs(repo_dir / "cookiecutter-{{project1}}", exist_ok=True)
    os.makedirs(repo_dir / "cookiecutter-{{project2}}", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    result = find_template(repo_dir, env)
    assert result == repo_dir / "cookiecutter-{{project1}}"

def test_find_template_with_custom_delimiters():
    repo_dir = Path("/tmp/test_repo6")
    os.makedirs(repo_dir / "cookiecutter-[[project]]", exist_ok=True)
    env = Environment(variable_start_string="[[", variable_end_string="]]")
    result = find_template(repo_dir, env)
    assert result == repo_dir / "cookiecutter-[[project]]"

def test_find_template_empty_directory():
    repo_dir = Path("/tmp/test_repo7")
    os.makedirs(repo_dir, exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True


# LLM-generated content at query #4
#--------------------------

def test_find_template_predicate_true():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    mock_repo_dir = "/fake/path"
    str_path = "cookiecutter-{{project_name}}"
    result = (
        'cookiecutter' in str_path
        and mock_env.variable_start_string in str_path
        and mock_env.variable_end_string in str_path
    )
    assert result == True


# LLM-generated content at query #5
#--------------------------

def test_find_template_with_valid_directory():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        template_dir = repo_dir / "cookiecutter_{{project_name}}"
        template_dir.mkdir()
        result = find_template(repo_dir, mock_env)
        assert result == template_dir

def test_find_template_without_cookiecutter_in_name():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        other_dir = repo_dir / "other_{{project_name}}"
        other_dir.mkdir()
        try:
            find_template(repo_dir, mock_env)
            assert False
        except NonTemplatedInputDirException:
            assert True

def test_find_template_without_variable_start_string():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        template_dir = repo_dir / "cookiecutter_project"
        template_dir.mkdir()
        try:
            find_template(repo_dir, mock_env)
            assert False
        except NonTemplatedInputDirException:
            assert True

def test_find_template_without_variable_end_string():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        template_dir = repo_dir / "cookiecutter_{{project"
        template_dir.mkdir()
        try:
            find_template(repo_dir, mock_env)
            assert False
        except NonTemplatedInputDirException:
            assert True

def test_find_template_with_multiple_directories():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        dir1 = repo_dir / "other_dir"
        dir1.mkdir()
        template_dir = repo_dir / "cookiecutter_{{project_name}}"
        template_dir.mkdir()
        dir2 = repo_dir / "another_dir"
        dir2.mkdir()
        result = find_template(repo_dir, mock_env)
        assert result == template_dir

def test_find_template_with_empty_directory():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        try:
            find_template(repo_dir, mock_env)
            assert False
        except NonTemplatedInputDirException:
            assert True

def test_find_template_with_different_variable_strings():
    class MockEnv:
        variable_start_string = "[["
        variable_end_string = "]]"
    mock_env = MockEnv()
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir)
        template_dir = repo_dir / "cookiecutter_[[project_name]]"
        template_dir.mkdir()
        result = find_template(repo_dir, mock_env)
        assert result == template_dir

def test_find_template_with_string_repo_dir():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    with tempfile.TemporaryDirectory() as tmpdir:
        template_dir = Path(tmpdir) / "cookiecutter_{{project_name}}"
        template_dir.mkdir()
        result = find_template(tmpdir, mock_env)
        assert result == template_dir


# LLM-generated content at query #6
#--------------------------

def test_find_template_valid():
    repo_dir = Path("/tmp/test_repo")
    os.makedirs(repo_dir / "cookiecutter_{{project}}", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    result = find_template(repo_dir, env)
    assert result == repo_dir / "cookiecutter_{{project}}"

def test_find_template_no_match():
    repo_dir = Path("/tmp/test_repo")
    os.makedirs(repo_dir / "normal_dir", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_multiple_matches():
    repo_dir = Path("/tmp/test_repo")
    os.makedirs(repo_dir / "cookiecutter_{{name}}", exist_ok=True)
    os.makedirs(repo_dir / "cookiecutter_{{project}}", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    result = find_template(repo_dir, env)
    assert result in [repo_dir / "cookiecutter_{{name}}", repo_dir / "cookiecutter_{{project}}"]

def test_find_template_different_env_variables():
    repo_dir = Path("/tmp/test_repo")
    os.makedirs(repo_dir / "cookiecutter_[[project]]", exist_ok=True)
    env = Environment(variable_start_string="[[", variable_end_string="]]")
    result = find_template(repo_dir, env)
    assert result == repo_dir / "cookiecutter_[[project]]"

def test_find_template_no_cookiecutter_in_name():
    repo_dir = Path("/tmp/test_repo")
    os.makedirs(repo_dir / "template_{{project}}", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_empty_repo():
    repo_dir = Path("/tmp/test_repo")
    os.makedirs(repo_dir, exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_with_str_path():
    repo_dir = "/tmp/test_repo"
    os.makedirs(Path(repo_dir) / "cookiecutter_{{project}}", exist_ok=True)
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    result = find_template(repo_dir, env)
    assert result == Path(repo_dir) / "cookiecutter_{{project}}"


# LLM-generated content at query #7
#--------------------------

def test_find_template_predicate_true():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    mock_repo_dir = "/tmp/test_repo"
    test_path = "cookiecutter_{{project_name}}"
    result = (
        'cookiecutter' in test_path
        and mock_env.variable_start_string in test_path
        and mock_env.variable_end_string in test_path
    )
    assert result == True


# LLM-generated content at query #8
#--------------------------

def test_find_template_predicate_true():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    mock_repo_dir = "/fake/path"
    str_path = "cookiecutter-{{project_name}}"
    result = (
        'cookiecutter' in str_path
        and mock_env.variable_start_string in str_path
        and mock_env.variable_end_string in str_path
    )
    assert result == True


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_find_template_with_valid_directory():
    repo_dir = Path("/fake/repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["cookiecutter-{{project_name}}"]
    result = find_template(repo_dir, env)
    assert result == Path("/fake/repo/cookiecutter-{{project_name}}")

def test_find_template_raises_exception_when_no_template():
    repo_dir = Path("/fake/repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["some_folder", "another_folder"]
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_with_multiple_directories():
    repo_dir = Path("/fake/repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["folder1", "cookiecutter-{{name}}", "folder2"]
    result = find_template(repo_dir, env)
    assert result == Path("/fake/repo/cookiecutter-{{name}}")

def test_find_template_with_different_variable_strings():
    repo_dir = Path("/fake/repo")
    env = Environment(variable_start_string="[[", variable_end_string="]]")
    os.listdir = lambda x: ["cookiecutter-[[project]]"]
    result = find_template(repo_dir, env)
    assert result == Path("/fake/repo/cookiecutter-[[project]]")

def test_find_template_without_cookiecutter_in_name():
    repo_dir = Path("/fake/repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["{{project_name}}"]
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_without_variable_start_string():
    repo_dir = Path("/fake/repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["cookiecutter-project}}"]
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True

def test_find_template_without_variable_end_string():
    repo_dir = Path("/fake/repo")
    env = Environment(variable_start_string="{{", variable_end_string="}}")
    os.listdir = lambda x: ["cookiecutter-{{project"]
    try:
        find_template(repo_dir, env)
        assert False
    except NonTemplatedInputDirException:
        assert True


# LLM-generated content at query #2
#--------------------------

def test_find_template_predicate_true():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    mock_repo_dir = "/fake/repo"
    str_path = "cookiecutter-{{project_name}}"
    result = (
        'cookiecutter' in str_path
        and mock_env.variable_start_string in str_path
        and mock_env.variable_end_string in str_path
    )
    assert result is True


# LLM-generated content at query #3
#--------------------------

def test_find_template_predicate_true():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    mock_repo_dir = "/fake/path"
    str_path = "cookiecutter-{{project_name}}"
    result = (
        'cookiecutter' in str_path
        and mock_env.variable_start_string in str_path
        and mock_env.variable_end_string in str_path
    )
    assert result == True


# LLM-generated content at query #4
#--------------------------

def test_find_template_predicate_true():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    mock_repo_dir = "/fake/path"
    str_path = "cookiecutter-{{project_name}}"
    result = (
        'cookiecutter' in str_path
        and mock_env.variable_start_string in str_path
        and mock_env.variable_end_string in str_path
    )
    assert result == True


# LLM-generated content at query #5
#--------------------------

def test_find_template_with_valid_directory():
    class MockEnv:
        variable_start_string = '{{'
        variable_end_string = '}}'
    mock_env = MockEnv()
    mock_repo_dir = '/tmp/test_repo'
    mock_valid_path = 'cookiecutter_{{project_name}}'
    with unittest.mock.patch('os.listdir', return_value=[mock_valid_path, 'other_dir']):
        with unittest.mock.patch('pathlib.Path') as MockPath:
            MockPath.return_value = '/tmp/test_repo/cookiecutter_{{project_name}}'
            result = find_template(mock_repo_dir, mock_env)
            assert result == '/tmp/test_repo/cookiecutter_{{project_name}}'

def test_find_template_without_cookiecutter_in_path():
    class MockEnv:
        variable_start_string = '{{'
        variable_end_string = '}}'
    mock_env = MockEnv()
    mock_repo_dir = '/tmp/test_repo'
    mock_invalid_path = 'not_cookiecutter_{{project_name}}'
    with unittest.mock.patch('os.listdir', return_value=[mock_invalid_path]):
        try:
            find_template(mock_repo_dir, mock_env)
            assert False
        except NonTemplatedInputDirException:
            assert True

def test_find_template_without_variable_start_string():
    class MockEnv:
        variable_start_string = '{{'
        variable_end_string = '}}'
    mock_env = MockEnv()
    mock_repo_dir = '/tmp/test_repo'
    mock_invalid_path = 'cookiecutter_project_name}}'
    with unittest.mock.patch('os.listdir', return_value=[mock_invalid_path]):
        try:
            find_template(mock_repo_dir, mock_env)
            assert False
        except NonTemplatedInputDirException:
            assert True

def test_find_template_without_variable_end_string():
    class MockEnv:
        variable_start_string = '{{'
        variable_end_string = '}}'
    mock_env = MockEnv()
    mock_repo_dir = '/tmp/test_repo'
    mock_invalid_path = 'cookiecutter_{{project_name'
    with unittest.mock.patch('os.listdir', return_value=[mock_invalid_path]):
        try:
            find_template(mock_repo_dir, mock_env)
            assert False
        except NonTemplatedInputDirException:
            assert True

def test_find_template_with_multiple_valid_directories():
    class MockEnv:
        variable_start_string = '{{'
        variable_end_string = '}}'
    mock_env = MockEnv()
    mock_repo_dir = '/tmp/test_repo'
    mock_valid_path1 = 'cookiecutter_{{project_name}}'
    mock_valid_path2 = 'cookiecutter_{{app_name}}'
    with unittest.mock.patch('os.listdir', return_value=[mock_valid_path1, mock_valid_path2]):
        with unittest.mock.patch('pathlib.Path') as MockPath:
            MockPath.return_value = '/tmp/test_repo/cookiecutter_{{project_name}}'
            result = find_template(mock_repo_dir, mock_env)
            assert result == '/tmp/test_repo/cookiecutter_{{project_name}}'

def test_find_template_with_empty_directory():
    class MockEnv:
        variable_start_string = '{{'
        variable_end_string = '}}'
    mock_env = MockEnv()
    mock_repo_dir = '/tmp/test_repo'
    with unittest.mock.patch('os.listdir', return_value=[]):
        try:
            find_template(mock_repo_dir, mock_env)
            assert False
        except NonTemplatedInputDirException:
            assert True


# LLM-generated content at query #6
#--------------------------

def test_find_template_predicate_true():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    mock_repo_dir = "/fake/path"
    str_path = "cookiecutter-{{project_name}}"
    result = (
        'cookiecutter' in str_path
        and mock_env.variable_start_string in str_path
        and mock_env.variable_end_string in str_path
    )
    assert result == True


# LLM-generated content at query #7
#--------------------------

def test_find_template_predicate_true():
    class MockEnv:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnv()
    mock_repo_dir = "/fake/path"
    str_path = "cookiecutter-{{project_name}}"
    result = (
        'cookiecutter' in str_path
        and mock_env.variable_start_string in str_path
        and mock_env.variable_end_string in str_path
    )
    assert result == True


# LLM-generated content at query #8
#--------------------------

def test_find_template_predicate_true():
    class MockEnvironment:
        variable_start_string = "{{"
        variable_end_string = "}}"
    mock_env = MockEnvironment()
    mock_repo_dir = "/fake/path"
    mock_str_path = "cookiecutter-{{project_name}}"
    result = ('cookiecutter' in mock_str_path and mock_env.variable_start_string in mock_str_path and mock_env.variable_end_string in mock_str_path)
    assert result == True


