# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.main as module_0
import importlib.metadata as module_1
import isort.settings as module_2
import isort.wrap_modes as module_3
import email.encoders as module_4
import urllib.parse as module_5

def test_case_0():
    pass

def test_case_1():
    var_0 = 'test_imports.py'
    var_1 = None
    var_2 = module_0.sort_imports(var_0, var_1, var_0)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.sort_imports(var_0, var_0, ask_to_apply=var_0, write_to_stdout=var_0)

def test_case_3():
    var_0 = []
    var_1 = module_0.main(var_0)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_4():
    var_0 = False
    var_1 = module_0.SortAttempt(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.main.SortAttempt'
    assert var_1.incorrectly_sorted is False
    assert var_1.skipped is False
    assert var_1.supported_encoding is False
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = []
    var_1 = module_0.main(var_0)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_2 = module_1.requires(var_1)
    module_0.sort_imports(var_2, var_1)

def test_case_6():
    var_0 = '--check'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_7():
    var_0 = '--dont-float-to-top'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_8():
    var_0 = '--dont-order-by-type'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_3 = bool('order_by_type' in var_2)
    assert var_3 is True
    var_4 = var_2['order_by_type']
    assert var_4 is False

def test_case_9():
    var_0 = '--dont-follow-links'
    var_1 = [var_0]
    var_2 = module_0.parse_args(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_10():
    var_0 = 'Vo72>>*Nyu!\x0br]y*Yu'
    var_1 = module_0.identify_imports_main(var_0)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_11():
    var_0 = False
    var_1 = 'color_output'
    var_2 = {var_1: var_0}
    var_3 = module_2.Config(**var_2)
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
    assert var_3.multi_line_output == module_3.WrapModes.GRID
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
    assert module_2.TYPE_CHECKING is False
    assert module_2.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_2.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_2.FIRSTPARTY == 'FIRSTPARTY'
    assert module_2.FUTURE == 'FUTURE'
    assert module_2.LOCALFOLDER == 'LOCALFOLDER'
    assert module_2.STDLIB == 'STDLIB'
    assert module_2.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_2.CYTHON_EXTENSIONS).__module__}.{type(module_2.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_2.SUPPORTED_EXTENSIONS).__module__}.{type(module_2.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_2.BLOCKED_EXTENSIONS).__module__}.{type(module_2.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.BLOCKED_EXTENSIONS) == 1
    assert module_2.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_2.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_2.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_2.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_2.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_2.DEFAULT_SKIP).__module__}.{type(module_2.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_SKIP) == 19
    assert module_2.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_2.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_2.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_2.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_2.KNOWN_PREFIX == 'known_'
    assert module_2.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_2.RUNTIME_SOURCE == 'runtime'
    assert module_2.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_2.DEFAULT_CONFIG).__module__}.{type(module_2.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_2.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_2.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_2.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.skip).__module__}.{type(module_2.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_2.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_2.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_2.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_2.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_2.DEFAULT_CONFIG.skip_gitignore is False
    assert module_2.DEFAULT_CONFIG.line_length == 79
    assert module_2.DEFAULT_CONFIG.wrap_length == 0
    assert module_2.DEFAULT_CONFIG.line_ending == ''
    assert module_2.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_2.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_2.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_2.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_2.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_2.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_2.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_2.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_2.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_2.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_2.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_2.DEFAULT_CONFIG.known_other == {}
    assert module_2.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_2.DEFAULT_CONFIG.forced_separate == ()
    assert module_2.DEFAULT_CONFIG.indent == '    '
    assert module_2.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_2.DEFAULT_CONFIG.length_sort is False
    assert module_2.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_2.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_2.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.add_imports).__module__}.{type(module_2.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_2.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.remove_imports) == 0
    assert module_2.DEFAULT_CONFIG.append_only is False
    assert module_2.DEFAULT_CONFIG.reverse_relative is False
    assert module_2.DEFAULT_CONFIG.force_single_line is False
    assert module_2.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_2.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_2.DEFAULT_CONFIG.import_headings == {}
    assert module_2.DEFAULT_CONFIG.import_footers == {}
    assert module_2.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_2.DEFAULT_CONFIG.use_parentheses is False
    assert module_2.DEFAULT_CONFIG.order_by_type is True
    assert module_2.DEFAULT_CONFIG.atomic is False
    assert module_2.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_2.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_2.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_2.DEFAULT_CONFIG.lines_between_types == 0
    assert module_2.DEFAULT_CONFIG.combine_as_imports is False
    assert module_2.DEFAULT_CONFIG.combine_star is False
    assert module_2.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_2.DEFAULT_CONFIG.from_first is False
    assert module_2.DEFAULT_CONFIG.verbose is False
    assert module_2.DEFAULT_CONFIG.quiet is False
    assert module_2.DEFAULT_CONFIG.force_adds is False
    assert module_2.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_2.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_2.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_2.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_2.DEFAULT_CONFIG.lexicographical is False
    assert module_2.DEFAULT_CONFIG.group_by_package is False
    assert module_2.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_2.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_2.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_2.DEFAULT_CONFIG.no_inline_sort is False
    assert module_2.DEFAULT_CONFIG.ignore_comments is False
    assert module_2.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_2.DEFAULT_CONFIG.sources).__module__}.{type(module_2.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_2.DEFAULT_CONFIG.sources) == 1
    assert module_2.DEFAULT_CONFIG.virtual_env == ''
    assert module_2.DEFAULT_CONFIG.conda_env == ''
    assert module_2.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
    assert module_2.DEFAULT_CONFIG.profile == ''
    assert module_2.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_2.DEFAULT_CONFIG.src_paths).__module__}.{type(module_2.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_2.DEFAULT_CONFIG.src_paths) == 2
    assert module_2.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_2.DEFAULT_CONFIG.float_to_top is False
    assert module_2.DEFAULT_CONFIG.filter_files is False
    assert module_2.DEFAULT_CONFIG.formatter == ''
    assert module_2.DEFAULT_CONFIG.formatting_function is None
    assert module_2.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_2.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_2.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_2.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_2.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_2.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_2.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_2.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_2.DEFAULT_CONFIG.constants).__module__}.{type(module_2.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.classes).__module__}.{type(module_2.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.variables).__module__}.{type(module_2.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.variables) == 0
    assert module_2.DEFAULT_CONFIG.dedup_headings is False
    assert module_2.DEFAULT_CONFIG.only_sections is False
    assert module_2.DEFAULT_CONFIG.only_modified is False
    assert module_2.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_2.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_2.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_2.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_2.DEFAULT_CONFIG.follow_links is True
    assert module_2.DEFAULT_CONFIG.indented_import_headings is True
    assert module_2.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_2.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_2.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_2.DEFAULT_CONFIG.reverse_sort is False
    assert module_2.DEFAULT_CONFIG.star_first is False
    assert module_2.DEFAULT_CONFIG.git_ls_files == {}
    assert module_2.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_2.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_2.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_2.DEFAULT_CONFIG.sort_reexports is False
    assert module_2.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_2.Config.known_patterns).__module__}.{type(module_2.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.section_comments).__module__}.{type(module_2.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.section_comments_end).__module__}.{type(module_2.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.skips).__module__}.{type(module_2.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.skip_globs).__module__}.{type(module_2.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.sorting_function).__module__}.{type(module_2.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_4 = 'Custom error message'
    var_5 = module_0._print_hard_fail(var_3, message=var_4)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_12():
    var_0 = '--version'
    var_1 = [var_0]
    var_2 = module_0.main(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_13():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--show-config'
    var_3 = '.'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.main(var_4)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '/nonexistent/path'
    var_1 = [var_0, var_0, var_0, var_0]
    module_0.main(var_1)

def test_case_15():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--show-files'
    var_3 = 'test.py'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = module_0.main(var_4)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = '--virtual-env'
    var_1 = '.'
    var_2 = '--show-files'
    var_3 = None
    var_4 = ''
    var_5 = [var_0, var_1, var_2, var_4]
    var_6 = module_0.main(var_5)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_7 = module_0.SortAttempt(var_3, var_3, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.main.SortAttempt'
    assert var_7.incorrectly_sorted is None
    assert var_7.skipped is None
    assert var_7.supported_encoding is None
    module_4.encode_7or8bit(var_7)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'g'
    var_1 = module_5.splittag(var_0)
    assert module_5.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_5.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_5.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_5.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_5.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_5.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_5.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_5.MAX_CACHE_SIZE == 20
    module_0.main(var_1)

def test_case_18():
    var_0 = '--multi-line=0'
    var_1 = [var_0, var_0]
    var_2 = module_0.parse_args(var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_3 = bool('multi_line_output' in var_2)
    assert var_3 is True

def test_case_19():
    var_0 = '--virtual-env'
    var_1 = '/nonexistent/path'
    var_2 = '--show-config'
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.main(var_3)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-lai', '-sg', '-wl', '-ws', '-ot', '-nis', '-lbt', '-fas', '-ca', '-fass', '-sp', '-fgw', '-ac', '-sl', '-sd', '-fss', '-af', '-ds', '-nlb', '-df', '-rr', '-tc', '-dt', '-le', '-cs', '-ff', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"