# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.main as module_0
import isort.settings as module_1
import isort.wrap_modes as module_2
import tokenize as module_3

def test_case_0():
    var_0 = None
    var_1 = module_0.SortAttempt(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.main.SortAttempt'
    assert var_1.incorrectly_sorted is None
    assert var_1.skipped is None
    assert var_1.supported_encoding is None
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.sort_imports(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = True
    module_0.sort_imports(var_0, var_0, var_1, write_to_stdout=var_0)

def test_case_3():
    var_0 = 'f\n2T/oYme'
    var_1 = None
    var_2 = module_0.sort_imports(var_0, var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_3 = bool('dont_float_to_top' not in var_2)
    assert var_3 is True
    var_4 = False
    module_0.sort_imports(var_3, var_3, var_4, var_3)

def test_case_5():
    var_0 = []
    var_1 = module_0.parse_args(var_0)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_6():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_7():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_3 = bool('dont_order_by_type' not in var_1)
    assert var_3 is True

def test_case_8():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_3 = bool('dont_follow_links' not in var_2)
    assert var_3 is True

def test_case_9():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_10():
    var_0 = []
    var_1 = module_0.main(var_0)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_11():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_1.Config(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.settings.Config'
    assert var_3.py_version == 'py3'
    assert f'{type(var_3.force_to_top).__module__}.{type(var_3.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.force_to_top) == 0
    assert f'{type(var_3.skip).__module__}.{type(var_3.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.skip) == 19
    assert f'{type(var_3.extend_skip).__module__}.{type(var_3.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.extend_skip) == 0
    assert f'{type(var_3.skip_glob).__module__}.{type(var_3.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.skip_glob) == 0
    assert f'{type(var_3.extend_skip_glob).__module__}.{type(var_3.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.extend_skip_glob) == 0
    assert var_3.skip_gitignore is False
    assert var_3.line_length == 79
    assert var_3.wrap_length == 0
    assert var_3.line_ending == ''
    assert var_3.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_3.no_sections is False
    assert f'{type(var_3.known_future_library).__module__}.{type(var_3.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_future_library) == 1
    assert f'{type(var_3.known_third_party).__module__}.{type(var_3.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_third_party) == 0
    assert f'{type(var_3.known_first_party).__module__}.{type(var_3.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_first_party) == 0
    assert f'{type(var_3.known_local_folder).__module__}.{type(var_3.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_local_folder) == 0
    assert f'{type(var_3.known_standard_library).__module__}.{type(var_3.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_standard_library) == 234
    assert f'{type(var_3.extra_standard_library).__module__}.{type(var_3.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.extra_standard_library) == 0
    assert var_3.known_other == {}
    assert var_3.multi_line_output == module_2.WrapModes.GRID
    assert var_3.forced_separate == ()
    assert var_3.indent == '    '
    assert var_3.comment_prefix == '  #'
    assert var_3.length_sort is False
    assert var_3.length_sort_straight is False
    assert f'{type(var_3.length_sort_sections).__module__}.{type(var_3.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.length_sort_sections) == 0
    assert f'{type(var_3.add_imports).__module__}.{type(var_3.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.add_imports) == 0
    assert f'{type(var_3.remove_imports).__module__}.{type(var_3.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.remove_imports) == 0
    assert var_3.append_only is False
    assert var_3.reverse_relative is False
    assert var_3.force_single_line is False
    assert var_3.single_line_exclusions == ()
    assert var_3.default_section == 'THIRDPARTY'
    assert var_3.import_headings == {}
    assert var_3.import_footers == {}
    assert var_3.balanced_wrapping is False
    assert var_3.use_parentheses is False
    assert var_3.order_by_type is True
    assert var_3.atomic is False
    assert var_3.lines_before_imports == -1
    assert var_3.lines_after_imports == -1
    assert var_3.lines_between_sections == 1
    assert var_3.lines_between_types == 0
    assert var_3.combine_as_imports is False
    assert var_3.combine_star is False
    assert var_3.include_trailing_comma is False
    assert var_3.from_first is False
    assert var_3.verbose is False
    assert var_3.quiet is False
    assert var_3.force_adds is False
    assert var_3.force_alphabetical_sort_within_sections is False
    assert var_3.force_alphabetical_sort is False
    assert var_3.force_grid_wrap == 0
    assert var_3.force_sort_within_sections is False
    assert var_3.lexicographical is False
    assert var_3.group_by_package is False
    assert var_3.ignore_whitespace is False
    assert f'{type(var_3.no_lines_before).__module__}.{type(var_3.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.no_lines_before) == 0
    assert var_3.no_inline_sort is False
    assert var_3.ignore_comments is False
    assert var_3.case_sensitive is False
    assert f'{type(var_3.sources).__module__}.{type(var_3.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_3.sources) == 2
    assert var_3.virtual_env == ''
    assert var_3.conda_env == ''
    assert var_3.ensure_newline_before_comments is False
    assert var_3.directory == '/workspace'
    assert var_3.profile == ''
    assert var_3.honor_noqa is False
    assert f'{type(var_3.src_paths).__module__}.{type(var_3.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_3.src_paths) == 2
    assert var_3.remove_redundant_aliases is False
    assert var_3.float_to_top is False
    assert var_3.filter_files is False
    assert var_3.formatter == ''
    assert var_3.formatting_function is None
    assert var_3.color_output is False
    assert f'{type(var_3.treat_comments_as_code).__module__}.{type(var_3.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.treat_comments_as_code) == 0
    assert var_3.treat_all_comments_as_code is False
    assert f'{type(var_3.supported_extensions).__module__}.{type(var_3.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.supported_extensions) == 4
    assert f'{type(var_3.blocked_extensions).__module__}.{type(var_3.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.blocked_extensions) == 1
    assert f'{type(var_3.constants).__module__}.{type(var_3.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.constants) == 0
    assert f'{type(var_3.classes).__module__}.{type(var_3.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.classes) == 0
    assert f'{type(var_3.variables).__module__}.{type(var_3.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.variables) == 0
    assert var_3.dedup_headings is False
    assert var_3.only_sections is False
    assert var_3.only_modified is False
    assert var_3.combine_straight_imports is False
    assert var_3.auto_identify_namespace_packages is True
    assert f'{type(var_3.namespace_packages).__module__}.{type(var_3.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.namespace_packages) == 0
    assert var_3.follow_links is True
    assert var_3.indented_import_headings is True
    assert var_3.honor_case_in_force_sorted_sections is False
    assert var_3.sort_relative_in_force_sorted_sections is False
    assert var_3.overwrite_in_place is False
    assert var_3.reverse_sort is False
    assert var_3.star_first is False
    assert var_3.git_ls_files == {}
    assert var_3.format_error == '{error}: {message}'
    assert var_3.format_success == '{success}: {message}'
    assert var_3.sort_order == 'natural'
    assert var_3.sort_reexports is False
    assert var_3.split_on_trailing_comma is False
    assert module_1.TYPE_CHECKING is False
    assert module_1.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_1.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.FIRSTPARTY == 'FIRSTPARTY'
    assert module_1.FUTURE == 'FUTURE'
    assert module_1.LOCALFOLDER == 'LOCALFOLDER'
    assert module_1.STDLIB == 'STDLIB'
    assert module_1.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_1.CYTHON_EXTENSIONS).__module__}.{type(module_1.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_1.SUPPORTED_EXTENSIONS).__module__}.{type(module_1.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_1.BLOCKED_EXTENSIONS).__module__}.{type(module_1.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.BLOCKED_EXTENSIONS) == 1
    assert module_1.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_1.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_1.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_1.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_1.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_1.DEFAULT_SKIP).__module__}.{type(module_1.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_SKIP) == 19
    assert module_1.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_1.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_1.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_1.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_1.KNOWN_PREFIX == 'known_'
    assert module_1.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_1.RUNTIME_SOURCE == 'runtime'
    assert module_1.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_1.Config.known_patterns).__module__}.{type(module_1.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.section_comments).__module__}.{type(module_1.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.section_comments_end).__module__}.{type(module_1.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.skips).__module__}.{type(module_1.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.skip_globs).__module__}.{type(module_1.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.sorting_function).__module__}.{type(module_1.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_4 = 'Custom error message'
    var_5 = 'file.py'
    var_6 = module_0._print_hard_fail(var_3, var_5, var_4)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = []
    var_1 = module_0.main(var_0)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-cs', '-sg', '-ot', '-sp', '-af', '-lbt', '-df', '-dt', '-ac', '-sl', '-nlb', '-fas', '-ds', '-tc', '-ls', '-sd', '-lai', '-ca', '-wl', '-fss', '-nis', '-le', '-ws', '-ff', '-fgw', '-rr', '-fass'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_2 = module_3.maybe(*var_0)
    assert var_2 == '()?'
    assert module_3.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_3.tok_name == {0: 'ENDMARKER', 1: 'NAME', 2: 'NUMBER', 3: 'STRING', 4: 'NEWLINE', 5: 'INDENT', 6: 'DEDENT', 7: 'LPAR', 8: 'RPAR', 9: 'LSQB', 10: 'RSQB', 11: 'COLON', 12: 'COMMA', 13: 'SEMI', 14: 'PLUS', 15: 'MINUS', 16: 'STAR', 17: 'SLASH', 18: 'VBAR', 19: 'AMPER', 20: 'LESS', 21: 'GREATER', 22: 'EQUAL', 23: 'DOT', 24: 'PERCENT', 25: 'LBRACE', 26: 'RBRACE', 27: 'EQEQUAL', 28: 'NOTEQUAL', 29: 'LESSEQUAL', 30: 'GREATEREQUAL', 31: 'TILDE', 32: 'CIRCUMFLEX', 33: 'LEFTSHIFT', 34: 'RIGHTSHIFT', 35: 'DOUBLESTAR', 36: 'PLUSEQUAL', 37: 'MINEQUAL', 38: 'STAREQUAL', 39: 'SLASHEQUAL', 40: 'PERCENTEQUAL', 41: 'AMPEREQUAL', 42: 'VBAREQUAL', 43: 'CIRCUMFLEXEQUAL', 44: 'LEFTSHIFTEQUAL', 45: 'RIGHTSHIFTEQUAL', 46: 'DOUBLESTAREQUAL', 47: 'DOUBLESLASH', 48: 'DOUBLESLASHEQUAL', 49: 'AT', 50: 'ATEQUAL', 51: 'RARROW', 52: 'ELLIPSIS', 53: 'COLONEQUAL', 54: 'OP', 55: 'AWAIT', 56: 'ASYNC', 57: 'TYPE_IGNORE', 58: 'TYPE_COMMENT', 59: 'SOFT_KEYWORD', 60: 'ERRORTOKEN', 61: 'COMMENT', 62: 'NL', 63: 'ENCODING', 64: 'N_TOKENS', 256: 'NT_OFFSET'}
    assert module_3.ENDMARKER == 0
    assert module_3.NAME == 1
    assert module_3.NUMBER == 2
    assert module_3.STRING == 3
    assert module_3.NEWLINE == 4
    assert module_3.INDENT == 5
    assert module_3.DEDENT == 6
    assert module_3.LPAR == 7
    assert module_3.RPAR == 8
    assert module_3.LSQB == 9
    assert module_3.RSQB == 10
    assert module_3.COLON == 11
    assert module_3.COMMA == 12
    assert module_3.SEMI == 13
    assert module_3.PLUS == 14
    assert module_3.MINUS == 15
    assert module_3.STAR == 16
    assert module_3.SLASH == 17
    assert module_3.VBAR == 18
    assert module_3.AMPER == 19
    assert module_3.LESS == 20
    assert module_3.GREATER == 21
    assert module_3.EQUAL == 22
    assert module_3.DOT == 23
    assert module_3.PERCENT == 24
    assert module_3.LBRACE == 25
    assert module_3.RBRACE == 26
    assert module_3.EQEQUAL == 27
    assert module_3.NOTEQUAL == 28
    assert module_3.LESSEQUAL == 29
    assert module_3.GREATEREQUAL == 30
    assert module_3.TILDE == 31
    assert module_3.CIRCUMFLEX == 32
    assert module_3.LEFTSHIFT == 33
    assert module_3.RIGHTSHIFT == 34
    assert module_3.DOUBLESTAR == 35
    assert module_3.PLUSEQUAL == 36
    assert module_3.MINEQUAL == 37
    assert module_3.STAREQUAL == 38
    assert module_3.SLASHEQUAL == 39
    assert module_3.PERCENTEQUAL == 40
    assert module_3.AMPEREQUAL == 41
    assert module_3.VBAREQUAL == 42
    assert module_3.CIRCUMFLEXEQUAL == 43
    assert module_3.LEFTSHIFTEQUAL == 44
    assert module_3.RIGHTSHIFTEQUAL == 45
    assert module_3.DOUBLESTAREQUAL == 46
    assert module_3.DOUBLESLASH == 47
    assert module_3.DOUBLESLASHEQUAL == 48
    assert module_3.AT == 49
    assert module_3.ATEQUAL == 50
    assert module_3.RARROW == 51
    assert module_3.ELLIPSIS == 52
    assert module_3.COLONEQUAL == 53
    assert module_3.OP == 54
    assert module_3.AWAIT == 55
    assert module_3.ASYNC == 56
    assert module_3.TYPE_IGNORE == 57
    assert module_3.TYPE_COMMENT == 58
    assert module_3.SOFT_KEYWORD == 59
    assert module_3.ERRORTOKEN == 60
    assert module_3.COMMENT == 61
    assert module_3.NL == 62
    assert module_3.ENCODING == 63
    assert module_3.N_TOKENS == 64
    assert module_3.NT_OFFSET == 256
    assert module_3.EXACT_TOKEN_TYPES == {'!=': 28, '%': 24, '%=': 40, '&': 19, '&=': 41, '(': 7, ')': 8, '*': 16, '**': 35, '**=': 46, '*=': 38, '+': 14, '+=': 36, ',': 12, '-': 15, '-=': 37, '->': 51, '.': 23, '...': 52, '/': 17, '//': 47, '//=': 48, '/=': 39, ':': 11, ':=': 53, ';': 13, '<': 20, '<<': 33, '<<=': 44, '<=': 29, '=': 22, '==': 27, '>': 21, '>=': 30, '>>': 34, '>>=': 45, '@': 49, '@=': 50, '[': 9, ']': 10, '^': 32, '^=': 43, '{': 25, '|': 18, '|=': 42, '}': 26, '~': 31}
    assert f'{type(module_3.cookie_re).__module__}.{type(module_3.cookie_re).__qualname__}' == 're.Pattern'
    assert f'{type(module_3.blank_re).__module__}.{type(module_3.blank_re).__qualname__}' == 're.Pattern'
    assert module_3.Whitespace == '[ \\f\\t]*'
    assert module_3.Comment == '#[^\\r\\n]*'
    assert module_3.Ignore == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?'
    assert module_3.Name == '\\w+'
    assert module_3.Hexnumber == '0[xX](?:_?[0-9a-fA-F])+'
    assert module_3.Binnumber == '0[bB](?:_?[01])+'
    assert module_3.Octnumber == '0[oO](?:_?[0-7])+'
    assert module_3.Decnumber == '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
    assert module_3.Intnumber == '(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*))'
    assert module_3.Exponent == '[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_3.Pointfloat == '([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?'
    assert module_3.Expfloat == '[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_3.Floatnumber == '(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)'
    assert module_3.Imagnumber == '([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])'
    assert module_3.Number == '(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))'
    assert module_3.StringPrefix == '(|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)'
    assert module_3.Single == "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
    assert module_3.Double == '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
    assert module_3.Single3 == "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
    assert module_3.Double3 == '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
    assert module_3.Triple == '((|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)\'\'\'|(|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)""")'
    assert module_3.String == '((|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")'
    assert module_3.Special == '(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=)'
    assert module_3.Funny == '(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))'
    assert module_3.PlainToken == '((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_3.Token == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_3.ContStr == '((|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))'
    assert module_3.PseudoExtras == '(\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)\'\'\'|(|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)"""))'
    assert module_3.PseudoToken == '[ \\f\\t]*((\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)\'\'\'|(|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)"""))|(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|RB|rB|fR|Rf|fr|Br|Rb|r|rf|bR|RF|B|f|R|Fr|U|BR|u|br|b|F|rF|rb|FR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))|\\w+)'
    assert module_3.endpats == {"'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", '"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", '"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "r'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'r"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "r'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'r"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "bR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'bR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "bR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'bR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "B'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'B"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "B'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'B"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "f'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'f"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "f'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'f"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "R'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'R"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "R'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'R"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "U'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'U"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "U'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'U"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "BR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'BR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "BR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'BR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "u'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'u"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "u'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'u"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "b'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'b"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "b'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'b"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "F'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'F"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "F'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'F"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "FR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'FR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "FR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'FR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'}
    assert module_3.single_quoted == {'Rf"', "rb'", '"', "r'", 'U"', "B'", "br'", "FR'", "U'", "Rf'", "b'", "RF'", 'rb"', 'F"', 'Fr"', 'R"', 'Br"', "R'", 'u"', "fR'", 'b"', 'r"', 'rB"', "BR'", 'RB"', "bR'", 'FR"', 'Rb"', 'fR"', 'BR"', "Fr'", "'", "rB'", 'bR"', 'rF"', "rF'", 'rf"', "f'", 'f"', "u'", 'fr"', "Br'", "fr'", "RB'", 'B"', 'br"', "Rb'", 'RF"', "rf'", "F'"}
    assert module_3.triple_quoted == {"bR'''", "'''", "Rf'''", 'rB"""', '"""', "RF'''", 'BR"""', 'FR"""', "FR'''", "Br'''", "RB'''", 'b"""', 'r"""', "rB'''", 'Rb"""', "F'''", "Rb'''", 'br"""', 'f"""', "f'''", "rb'''", 'Fr"""', "br'''", "U'''", 'fR"""', 'F"""', 'Rf"""', "fR'''", 'B"""', "rF'''", "b'''", "u'''", 'bR"""', "B'''", 'RF"""', 'u"""', 'Br"""', 'rf"""', "fr'''", "r'''", 'RB"""', 'rF"""', "BR'''", "Fr'''", 'rb"""', 'fr"""', "rf'''", 'R"""', "R'''", 'U"""'}
    assert module_3.t == 'FR'
    assert module_3.u == "FR'''"
    assert module_3.tabsize == 8
    var_3 = True
    var_4 = module_0.identify_imports_main(var_2)
    var_5 = 'KZ>dCUXj/'
    var_6 = var_2.__le__(var_4)
    var_7 = module_0.sort_imports(var_5, var_2, var_2)
    var_8 = module_0.parse_args(var_2)
    module_0.sort_imports(var_1, var_2, var_1, write_to_stdout=var_3)