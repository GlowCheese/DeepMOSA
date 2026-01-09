####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_path_initialization_with_default_platform():
    path = Path()
    assert path.platform == sys.platform


def test_path_initialization_with_specific_platform():
    path = Path(platform="linux")
    assert path.platform == "linux"


def test_path_initialization_with_win32_platform():
    path = Path(platform="win32")
    assert path.platform == "win32"
    assert isinstance(path._pathlib_home, PureWindowsPath)


def test_path_initialization_with_darwin_platform():
    path = Path(platform="darwin")
    assert path.platform == "darwin"
    assert isinstance(path._pathlib_home, PurePosixPath)


def test_path_initialization_with_freebsd_variant():
    path = Path(platform="freebsd10")
    assert path.platform == "freebsd"


def test_path_initialization_with_win64_platform():
    path = Path(platform="win64")
    assert path.platform == "win64"
    assert isinstance(path._pathlib_home, PureWindowsPath)


def test_path_initialization_with_seed():
    path = Path(seed=42)
    assert path.seed == 42


def test_path_initialization_with_random_instance():
    custom_random = _random.Random()
    path = Path(random=custom_random)
    assert path.random is custom_random


def test_path_initialization_with_platform_and_seed():
    path = Path(platform="linux", seed=123)
    assert path.platform == "linux"
    assert path.seed == 123


def test_path_initialization_home_path_for_linux():
    path = Path(platform="linux")
    expected_home = PurePosixPath() / PLATFORMS["linux"]["home"]
    assert str(path._pathlib_home) == str(expected_home)


def test_path_initialization_home_path_for_win32():
    path = Path(platform="win32")
    expected_home = PureWindowsPath() / PLATFORMS["win32"]["home"]
    assert str(path._pathlib_home) == str(expected_home)


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_path_initialization_with_default_platform():
    provider = Path()
    assert provider.platform == sys.platform


def test_path_initialization_with_specific_platform():
    provider = Path(platform="linux")
    assert provider.platform == "linux"


def test_path_initialization_with_freebsd_variant():
    provider = Path(platform="freebsd10")
    assert provider.platform == "freebsd"


def test_path_initialization_with_windows_platform():
    provider = Path(platform="win32")
    assert provider.platform == "win32"
    assert isinstance(provider._pathlib_home, PureWindowsPath)


def test_path_initialization_with_posix_platform():
    provider = Path(platform="darwin")
    assert provider.platform == "darwin"
    assert isinstance(provider._pathlib_home, PurePosixPath)


def test_path_initialization_with_seed():
    provider = Path(seed=42)
    assert provider.seed == 42


def test_path_initialization_with_random_instance():
    custom_random = _random.Random()
    provider = Path(random=custom_random)
    assert provider.random is custom_random


def test_path_initialization_with_platform_and_seed():
    provider = Path(platform="linux", seed=123)
    assert provider.platform == "linux"
    assert provider.seed == 123


def test_path_initialization_home_path_set_correctly():
    provider = Path(platform="linux")
    expected_home = PLATFORMS["linux"]["home"]
    assert str(provider._pathlib_home) == expected_home


