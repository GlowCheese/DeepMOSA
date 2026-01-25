####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_path_initialization_with_default_platform():
    provider = Path()
    assert provider.platform == sys.platform
    assert isinstance(provider._pathlib_home, PurePosixPath) if not sys.platform.startswith('win') else isinstance(provider._pathlib_home, PureWindowsPath)


def test_path_initialization_with_specific_platform_linux():
    provider = Path(platform='linux')
    assert provider.platform == 'linux'
    assert isinstance(provider._pathlib_home, PurePosixPath)
    assert str(provider._pathlib_home) == PLATFORMS['linux']['home']


def test_path_initialization_with_specific_platform_darwin():
    provider = Path(platform='darwin')
    assert provider.platform == 'darwin'
    assert isinstance(provider._pathlib_home, PurePosixPath)
    assert str(provider._pathlib_home) == PLATFORMS['darwin']['home']


def test_path_initialization_with_specific_platform_win32():
    provider = Path(platform='win32')
    assert provider.platform == 'win32'
    assert isinstance(provider._pathlib_home, PureWindowsPath)
    assert str(provider._pathlib_home) == PLATFORMS['win32']['home']


def test_path_initialization_with_specific_platform_win64():
    provider = Path(platform='win64')
    assert provider.platform == 'win64'
    assert isinstance(provider._pathlib_home, PureWindowsPath)
    assert str(provider._pathlib_home) == PLATFORMS['win64']['home']


def test_path_initialization_with_specific_platform_freebsd():
    provider = Path(platform='freebsd10')
    assert provider.platform == 'freebsd'
    assert isinstance(provider._pathlib_home, PurePosixPath)
    assert str(provider._pathlib_home) == PLATFORMS['freebsd']['home']


def test_path_initialization_with_seed():
    seed = 12345
    provider = Path(seed=seed)
    assert provider.seed == seed


def test_path_initialization_with_random_instance():
    custom_random = _random.Random()
    provider = Path(random=custom_random)
    assert provider.random is custom_random


def test_path_initialization_with_invalid_random_type():
    try:
        Path(random="invalid")
        assert False
    except TypeError:
        assert True


def test_path_initialization_with_keyword_only_arguments():
    provider = Path(seed=999, platform='linux')
    assert provider.seed == 999
    assert provider.platform == 'linux'


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_user_returns_path_with_username():
    from mimesis import Path
    path = Path(platform="linux")
    result = path.user()
    assert result.startswith("/home/")
    assert result.count("/") == 2

def test_user_returns_capitalized_username_on_windows():
    from mimesis import Path
    path = Path(platform="win32")
    result = path.user()
    assert result.startswith("C:\\Users\\")
    username_part = result.split("\\")[2]
    assert username_part[0].isupper()

def test_user_returns_lowercase_username_on_linux():
    from mimesis import Path
    path = Path(platform="linux")
    result = path.user()
    username_part = result.split("/")[2]
    assert username_part.islower()

def test_user_returns_valid_username_from_list():
    from mimesis import Path
    from mimesis.builtins import USERNAMES
    path = Path(platform="linux")
    result = path.user()
    username_part = result.split("/")[2]
    assert username_part in [u.lower() for u in USERNAMES]

def test_user_returns_different_usernames_on_multiple_calls():
    from mimesis import Path
    path = Path(platform="linux")
    results = [path.user() for _ in range(10)]
    usernames = [r.split("/")[2] for r in results]
    assert len(set(usernames)) > 1


# LLM-generated content at query #2
#--------------------------

def test_path_initialization_with_default_platform():
    path = Path()
    assert path.platform == sys.platform


def test_path_initialization_with_specific_platform():
    path = Path(platform='linux')
    assert path.platform == 'linux'


def test_path_initialization_with_win32_platform():
    path = Path(platform='win32')
    assert path.platform == 'win32'
    assert isinstance(path._pathlib_home, PureWindowsPath)


def test_path_initialization_with_darwin_platform():
    path = Path(platform='darwin')
    assert path.platform == 'darwin'
    assert isinstance(path._pathlib_home, PurePosixPath)


def test_path_initialization_with_freebsd_variant():
    path = Path(platform='freebsd10')
    assert path.platform == 'freebsd'


def test_path_initialization_with_seed():
    path = Path(seed=42)
    assert path.seed == 42


def test_path_initialization_with_random_instance():
    custom_random = _random.Random()
    path = Path(random=custom_random)
    assert path.random is custom_random


def test_path_initialization_with_platform_and_seed():
    path = Path(platform='linux', seed=123)
    assert path.platform == 'linux'
    assert path.seed == 123


def test_path_initialization_with_platform_and_random():
    custom_random = _random.Random()
    path = Path(platform='win64', random=custom_random)
    assert path.platform == 'win64'
    assert path.random is custom_random


def test_path_initialization_with_invalid_random_type():
    try:
        Path(random='invalid')
        assert False
    except TypeError:
        assert True


