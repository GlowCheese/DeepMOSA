# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.settings as module_0
import isort.wrap_modes as module_1
import importlib.metadata as module_2
import enum as module_3
import isort.exceptions as module_4

def test_case_0():
    pass

def test_case_1():
    var_0 = ',d'
    var_1 = module_0.find_all_configs(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False

def test_case_2():
    var_0 = None
    var_1 = module_0.entry_points(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'importlib.metadata.EntryPoints'
    assert len(var_1) == 0
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_2.EntryPoints.names).__module__}.{type(module_2.EntryPoints.names).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.EntryPoints.groups).__module__}.{type(module_2.EntryPoints.groups).__qualname__}' == 'builtins.property'

def test_case_3():
    var_0 = 'project'
    var_1 = module_0.Config(settings_path=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.settings.Config'
    assert var_1.py_version == 'py3'
    assert f'{type(var_1.force_to_top).__module__}.{type(var_1.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.force_to_top) == 0
    assert f'{type(var_1.skip).__module__}.{type(var_1.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.skip) == 1
    assert f'{type(var_1.extend_skip).__module__}.{type(var_1.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extend_skip) == 0
    assert f'{type(var_1.skip_glob).__module__}.{type(var_1.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.skip_glob) == 0
    assert f'{type(var_1.extend_skip_glob).__module__}.{type(var_1.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extend_skip_glob) == 0
    assert var_1.skip_gitignore is False
    assert var_1.line_length == 100
    assert var_1.wrap_length == 0
    assert var_1.line_ending == ''
    assert var_1.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_1.no_sections is False
    assert f'{type(var_1.known_future_library).__module__}.{type(var_1.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_future_library) == 1
    assert f'{type(var_1.known_third_party).__module__}.{type(var_1.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_third_party) == 0
    assert f'{type(var_1.known_first_party).__module__}.{type(var_1.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_first_party) == 0
    assert f'{type(var_1.known_local_folder).__module__}.{type(var_1.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_local_folder) == 0
    assert f'{type(var_1.known_standard_library).__module__}.{type(var_1.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_standard_library) == 234
    assert f'{type(var_1.extra_standard_library).__module__}.{type(var_1.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extra_standard_library) == 0
    assert var_1.known_other == {}
    assert var_1.multi_line_output == module_1.WrapModes.VERTICAL_HANGING_INDENT
    assert var_1.forced_separate == ()
    assert var_1.indent == '    '
    assert var_1.comment_prefix == '  #'
    assert var_1.length_sort is False
    assert var_1.length_sort_straight is False
    assert f'{type(var_1.length_sort_sections).__module__}.{type(var_1.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.length_sort_sections) == 0
    assert f'{type(var_1.add_imports).__module__}.{type(var_1.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.add_imports) == 0
    assert f'{type(var_1.remove_imports).__module__}.{type(var_1.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.remove_imports) == 0
    assert var_1.append_only is False
    assert var_1.reverse_relative is False
    assert var_1.force_single_line is False
    assert var_1.single_line_exclusions == ()
    assert var_1.default_section == 'THIRDPARTY'
    assert var_1.import_headings == {}
    assert var_1.import_footers == {}
    assert var_1.balanced_wrapping is False
    assert var_1.use_parentheses is True
    assert var_1.order_by_type is True
    assert var_1.atomic is False
    assert var_1.lines_before_imports == -1
    assert var_1.lines_after_imports == -1
    assert var_1.lines_between_sections == 1
    assert var_1.lines_between_types == 0
    assert var_1.combine_as_imports is False
    assert var_1.combine_star is False
    assert var_1.include_trailing_comma is True
    assert var_1.from_first is False
    assert var_1.verbose is False
    assert var_1.quiet is False
    assert var_1.force_adds is False
    assert var_1.force_alphabetical_sort_within_sections is False
    assert var_1.force_alphabetical_sort is False
    assert var_1.force_grid_wrap == 0
    assert var_1.force_sort_within_sections is False
    assert var_1.lexicographical is False
    assert var_1.group_by_package is False
    assert var_1.ignore_whitespace is False
    assert f'{type(var_1.no_lines_before).__module__}.{type(var_1.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.no_lines_before) == 0
    assert var_1.no_inline_sort is False
    assert var_1.ignore_comments is False
    assert var_1.case_sensitive is False
    assert f'{type(var_1.sources).__module__}.{type(var_1.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_1.sources) == 3
    assert var_1.virtual_env == ''
    assert var_1.conda_env == ''
    assert var_1.ensure_newline_before_comments is False
    assert var_1.directory == '/workspace/project'
    assert var_1.profile == 'hug'
    assert var_1.honor_noqa is False
    assert f'{type(var_1.src_paths).__module__}.{type(var_1.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_1.src_paths) == 2
    assert var_1.remove_redundant_aliases is False
    assert var_1.float_to_top is False
    assert var_1.filter_files is False
    assert var_1.formatter == ''
    assert var_1.formatting_function is None
    assert var_1.color_output is False
    assert f'{type(var_1.treat_comments_as_code).__module__}.{type(var_1.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.treat_comments_as_code) == 0
    assert var_1.treat_all_comments_as_code is False
    assert f'{type(var_1.supported_extensions).__module__}.{type(var_1.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.supported_extensions) == 4
    assert f'{type(var_1.blocked_extensions).__module__}.{type(var_1.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.blocked_extensions) == 1
    assert f'{type(var_1.constants).__module__}.{type(var_1.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.constants) == 0
    assert f'{type(var_1.classes).__module__}.{type(var_1.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.classes) == 0
    assert f'{type(var_1.variables).__module__}.{type(var_1.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.variables) == 0
    assert var_1.dedup_headings is False
    assert var_1.only_sections is False
    assert var_1.only_modified is False
    assert var_1.combine_straight_imports is False
    assert var_1.auto_identify_namespace_packages is True
    assert f'{type(var_1.namespace_packages).__module__}.{type(var_1.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.namespace_packages) == 0
    assert var_1.follow_links is True
    assert var_1.indented_import_headings is True
    assert var_1.honor_case_in_force_sorted_sections is False
    assert var_1.sort_relative_in_force_sorted_sections is False
    assert var_1.overwrite_in_place is False
    assert var_1.reverse_sort is False
    assert var_1.star_first is False
    assert var_1.git_ls_files == {}
    assert var_1.format_error == '{error}: {message}'
    assert var_1.format_success == '{success}: {message}'
    assert var_1.sort_order == 'natural'
    assert var_1.sort_reexports is False
    assert var_1.split_on_trailing_comma is False
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_0.Config.known_patterns).__module__}.{type(module_0.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments).__module__}.{type(module_0.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments_end).__module__}.{type(module_0.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skips).__module__}.{type(module_0.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skip_globs).__module__}.{type(module_0.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.sorting_function).__module__}.{type(module_0.Config.sorting_function).__qualname__}' == 'builtins.property'

def test_case_4():
    var_0 = module_3._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = 'G|E\rp_)0Kp OH<='
    with pytest.raises(module_4.InvalidSettingsPath):
        module_0.Config(settings_path=var_1)

def test_case_5():
    var_0 = 'oyA\t"M'
    var_1 = module_0.find_all_configs(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    var_2 = '~]oRnQi3B'
    var_3 = var_1.search(var_2)
    var_4 = module_0.Config()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.settings.Config'
    assert var_4.py_version == 'py3'
    assert f'{type(var_4.force_to_top).__module__}.{type(var_4.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.force_to_top) == 0
    assert f'{type(var_4.skip).__module__}.{type(var_4.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.skip) == 19
    assert f'{type(var_4.extend_skip).__module__}.{type(var_4.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.extend_skip) == 0
    assert f'{type(var_4.skip_glob).__module__}.{type(var_4.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.skip_glob) == 0
    assert f'{type(var_4.extend_skip_glob).__module__}.{type(var_4.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.extend_skip_glob) == 0
    assert var_4.skip_gitignore is False
    assert var_4.line_length == 79
    assert var_4.wrap_length == 0
    assert var_4.line_ending == ''
    assert var_4.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_4.no_sections is False
    assert f'{type(var_4.known_future_library).__module__}.{type(var_4.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_future_library) == 1
    assert f'{type(var_4.known_third_party).__module__}.{type(var_4.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_third_party) == 0
    assert f'{type(var_4.known_first_party).__module__}.{type(var_4.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_first_party) == 0
    assert f'{type(var_4.known_local_folder).__module__}.{type(var_4.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_local_folder) == 0
    assert f'{type(var_4.known_standard_library).__module__}.{type(var_4.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_standard_library) == 234
    assert f'{type(var_4.extra_standard_library).__module__}.{type(var_4.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.extra_standard_library) == 0
    assert var_4.known_other == {}
    assert var_4.multi_line_output == module_1.WrapModes.GRID
    assert var_4.forced_separate == ()
    assert var_4.indent == '    '
    assert var_4.comment_prefix == '  #'
    assert var_4.length_sort is False
    assert var_4.length_sort_straight is False
    assert f'{type(var_4.length_sort_sections).__module__}.{type(var_4.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.length_sort_sections) == 0
    assert f'{type(var_4.add_imports).__module__}.{type(var_4.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.add_imports) == 0
    assert f'{type(var_4.remove_imports).__module__}.{type(var_4.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.remove_imports) == 0
    assert var_4.append_only is False
    assert var_4.reverse_relative is False
    assert var_4.force_single_line is False
    assert var_4.single_line_exclusions == ()
    assert var_4.default_section == 'THIRDPARTY'
    assert var_4.import_headings == {}
    assert var_4.import_footers == {}
    assert var_4.balanced_wrapping is False
    assert var_4.use_parentheses is False
    assert var_4.order_by_type is True
    assert var_4.atomic is False
    assert var_4.lines_before_imports == -1
    assert var_4.lines_after_imports == -1
    assert var_4.lines_between_sections == 1
    assert var_4.lines_between_types == 0
    assert var_4.combine_as_imports is False
    assert var_4.combine_star is False
    assert var_4.include_trailing_comma is False
    assert var_4.from_first is False
    assert var_4.verbose is False
    assert var_4.quiet is False
    assert var_4.force_adds is False
    assert var_4.force_alphabetical_sort_within_sections is False
    assert var_4.force_alphabetical_sort is False
    assert var_4.force_grid_wrap == 0
    assert var_4.force_sort_within_sections is False
    assert var_4.lexicographical is False
    assert var_4.group_by_package is False
    assert var_4.ignore_whitespace is False
    assert f'{type(var_4.no_lines_before).__module__}.{type(var_4.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.no_lines_before) == 0
    assert var_4.no_inline_sort is False
    assert var_4.ignore_comments is False
    assert var_4.case_sensitive is False
    assert f'{type(var_4.sources).__module__}.{type(var_4.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_4.sources) == 1
    assert var_4.virtual_env == ''
    assert var_4.conda_env == ''
    assert var_4.ensure_newline_before_comments is False
    assert var_4.directory == '/workspace'
    assert var_4.profile == ''
    assert var_4.honor_noqa is False
    assert f'{type(var_4.src_paths).__module__}.{type(var_4.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_4.src_paths) == 2
    assert var_4.remove_redundant_aliases is False
    assert var_4.float_to_top is False
    assert var_4.filter_files is False
    assert var_4.formatter == ''
    assert var_4.formatting_function is None
    assert var_4.color_output is False
    assert f'{type(var_4.treat_comments_as_code).__module__}.{type(var_4.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.treat_comments_as_code) == 0
    assert var_4.treat_all_comments_as_code is False
    assert f'{type(var_4.supported_extensions).__module__}.{type(var_4.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.supported_extensions) == 4
    assert f'{type(var_4.blocked_extensions).__module__}.{type(var_4.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.blocked_extensions) == 1
    assert f'{type(var_4.constants).__module__}.{type(var_4.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.constants) == 0
    assert f'{type(var_4.classes).__module__}.{type(var_4.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.classes) == 0
    assert f'{type(var_4.variables).__module__}.{type(var_4.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.variables) == 0
    assert var_4.dedup_headings is False
    assert var_4.only_sections is False
    assert var_4.only_modified is False
    assert var_4.combine_straight_imports is False
    assert var_4.auto_identify_namespace_packages is True
    assert f'{type(var_4.namespace_packages).__module__}.{type(var_4.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.namespace_packages) == 0
    assert var_4.follow_links is True
    assert var_4.indented_import_headings is True
    assert var_4.honor_case_in_force_sorted_sections is False
    assert var_4.sort_relative_in_force_sorted_sections is False
    assert var_4.overwrite_in_place is False
    assert var_4.reverse_sort is False
    assert var_4.star_first is False
    assert var_4.git_ls_files == {}
    assert var_4.format_error == '{error}: {message}'
    assert var_4.format_success == '{success}: {message}'
    assert var_4.sort_order == 'natural'
    assert var_4.sort_reexports is False
    assert var_4.split_on_trailing_comma is False
    assert f'{type(module_0.Config.known_patterns).__module__}.{type(module_0.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments).__module__}.{type(module_0.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments_end).__module__}.{type(module_0.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skips).__module__}.{type(module_0.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skip_globs).__module__}.{type(module_0.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.sorting_function).__module__}.{type(module_0.Config.sorting_function).__qualname__}' == 'builtins.property'
    with pytest.raises(ValueError):
        var_4.__post_init__()

def test_case_6():
    var_0 = None
    var_1 = {}
    var_2 = False
    var_3 = [var_1]
    var_4 = 'fM/YZGh\r@L=VAOxa'
    var_5 = -5011
    var_6 = module_0._Config(force_to_top=var_0, skip=var_0, extend_skip=var_0, extend_skip_glob=var_1, wrap_length=var_2, known_standard_library=var_3, comment_prefix=var_4, length_sort=var_0, length_sort_sections=var_0, reverse_relative=var_0, import_footers=var_0, lines_after_imports=var_5, lines_between_sections=var_0, verbose=var_0, group_by_package=var_4, no_lines_before=var_0, virtual_env=var_0, profile=var_4, honor_noqa=var_5, remove_redundant_aliases=var_4, filter_files=var_2, constants=var_5, variables=var_0, only_modified=var_0, namespace_packages=var_0, indented_import_headings=var_4, overwrite_in_place=var_0, reverse_sort=var_0, star_first=var_0, sort_order=var_4, sort_reexports=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.settings._Config'
    assert var_6.py_version == 'py3'
    assert var_6.force_to_top is None
    assert var_6.skip is None
    assert var_6.extend_skip is None
    assert f'{type(var_6.skip_glob).__module__}.{type(var_6.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.skip_glob) == 0
    assert var_6.extend_skip_glob == {}
    assert var_6.skip_gitignore is False
    assert var_6.line_length == 79
    assert var_6.wrap_length is False
    assert var_6.line_ending == ''
    assert var_6.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_6.no_sections is False
    assert f'{type(var_6.known_future_library).__module__}.{type(var_6.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.known_future_library) == 1
    assert f'{type(var_6.known_third_party).__module__}.{type(var_6.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.known_third_party) == 0
    assert f'{type(var_6.known_first_party).__module__}.{type(var_6.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.known_first_party) == 0
    assert f'{type(var_6.known_local_folder).__module__}.{type(var_6.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.known_local_folder) == 0
    assert var_6.known_standard_library == [{}]
    assert f'{type(var_6.extra_standard_library).__module__}.{type(var_6.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.extra_standard_library) == 0
    assert var_6.known_other == {}
    assert var_6.multi_line_output == module_1.WrapModes.GRID
    assert var_6.forced_separate == ()
    assert var_6.indent == '    '
    assert var_6.comment_prefix == 'fM/YZGh\r@L=VAOxa'
    assert var_6.length_sort is None
    assert var_6.length_sort_straight is False
    assert var_6.length_sort_sections is None
    assert f'{type(var_6.add_imports).__module__}.{type(var_6.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.add_imports) == 0
    assert f'{type(var_6.remove_imports).__module__}.{type(var_6.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.remove_imports) == 0
    assert var_6.append_only is False
    assert var_6.reverse_relative is None
    assert var_6.force_single_line is False
    assert var_6.single_line_exclusions == ()
    assert var_6.default_section == 'THIRDPARTY'
    assert var_6.import_headings == {}
    assert var_6.import_footers is None
    assert var_6.balanced_wrapping is False
    assert var_6.use_parentheses is False
    assert var_6.order_by_type is True
    assert var_6.atomic is False
    assert var_6.lines_before_imports == -1
    assert var_6.lines_after_imports == -5011
    assert var_6.lines_between_sections is None
    assert var_6.lines_between_types == 0
    assert var_6.combine_as_imports is False
    assert var_6.combine_star is False
    assert var_6.include_trailing_comma is False
    assert var_6.from_first is False
    assert var_6.verbose is None
    assert var_6.quiet is False
    assert var_6.force_adds is False
    assert var_6.force_alphabetical_sort_within_sections is False
    assert var_6.force_alphabetical_sort is False
    assert var_6.force_grid_wrap == 0
    assert var_6.force_sort_within_sections is False
    assert var_6.lexicographical is False
    assert var_6.group_by_package == 'fM/YZGh\r@L=VAOxa'
    assert var_6.ignore_whitespace is False
    assert var_6.no_lines_before is None
    assert var_6.no_inline_sort is False
    assert var_6.ignore_comments is False
    assert var_6.case_sensitive is False
    assert var_6.sources == ()
    assert var_6.virtual_env is None
    assert var_6.conda_env == ''
    assert var_6.ensure_newline_before_comments is False
    assert var_6.directory == ''
    assert var_6.profile == 'fM/YZGh\r@L=VAOxa'
    assert var_6.honor_noqa == -5011
    assert var_6.src_paths == ()
    assert var_6.remove_redundant_aliases == 'fM/YZGh\r@L=VAOxa'
    assert var_6.float_to_top is False
    assert var_6.filter_files is False
    assert var_6.formatter == ''
    assert var_6.formatting_function is None
    assert var_6.color_output is False
    assert f'{type(var_6.treat_comments_as_code).__module__}.{type(var_6.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.treat_comments_as_code) == 0
    assert var_6.treat_all_comments_as_code is False
    assert f'{type(var_6.supported_extensions).__module__}.{type(var_6.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.supported_extensions) == 4
    assert f'{type(var_6.blocked_extensions).__module__}.{type(var_6.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.blocked_extensions) == 1
    assert var_6.constants == -5011
    assert f'{type(var_6.classes).__module__}.{type(var_6.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.classes) == 0
    assert var_6.variables is None
    assert var_6.dedup_headings is False
    assert var_6.only_sections is False
    assert var_6.only_modified is None
    assert var_6.combine_straight_imports is False
    assert var_6.auto_identify_namespace_packages is True
    assert var_6.namespace_packages is None
    assert var_6.follow_links is True
    assert var_6.indented_import_headings == 'fM/YZGh\r@L=VAOxa'
    assert var_6.honor_case_in_force_sorted_sections is False
    assert var_6.sort_relative_in_force_sorted_sections is False
    assert var_6.overwrite_in_place is None
    assert var_6.reverse_sort is None
    assert var_6.star_first is None
    assert var_6.git_ls_files == {}
    assert var_6.format_error == '{error}: {message}'
    assert var_6.format_success == '{success}: {message}'
    assert var_6.sort_order == 'fM/YZGh\r@L=VAOxa'
    assert var_6.sort_reexports is False
    assert var_6.split_on_trailing_comma is False
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert module_0._Config.py_version == '3'
    assert f'{type(module_0._Config.force_to_top).__module__}.{type(module_0._Config.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.force_to_top) == 0
    assert f'{type(module_0._Config.skip).__module__}.{type(module_0._Config.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.skip) == 19
    assert f'{type(module_0._Config.extend_skip).__module__}.{type(module_0._Config.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.extend_skip) == 0
    assert f'{type(module_0._Config.skip_glob).__module__}.{type(module_0._Config.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.skip_glob) == 0
    assert f'{type(module_0._Config.extend_skip_glob).__module__}.{type(module_0._Config.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.extend_skip_glob) == 0
    assert module_0._Config.skip_gitignore is False
    assert module_0._Config.line_length == 79
    assert module_0._Config.wrap_length == 0
    assert module_0._Config.line_ending == ''
    assert module_0._Config.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0._Config.no_sections is False
    assert f'{type(module_0._Config.known_future_library).__module__}.{type(module_0._Config.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_future_library) == 1
    assert f'{type(module_0._Config.known_third_party).__module__}.{type(module_0._Config.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_third_party) == 0
    assert f'{type(module_0._Config.known_first_party).__module__}.{type(module_0._Config.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_first_party) == 0
    assert f'{type(module_0._Config.known_local_folder).__module__}.{type(module_0._Config.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_local_folder) == 0
    assert f'{type(module_0._Config.known_standard_library).__module__}.{type(module_0._Config.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_standard_library) == 0
    assert f'{type(module_0._Config.extra_standard_library).__module__}.{type(module_0._Config.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.extra_standard_library) == 0
    assert module_0._Config.multi_line_output == module_1.WrapModes.GRID
    assert module_0._Config.forced_separate == ()
    assert module_0._Config.indent == '    '
    assert module_0._Config.comment_prefix == '  #'
    assert module_0._Config.length_sort is False
    assert module_0._Config.length_sort_straight is False
    assert f'{type(module_0._Config.length_sort_sections).__module__}.{type(module_0._Config.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.length_sort_sections) == 0
    assert f'{type(module_0._Config.add_imports).__module__}.{type(module_0._Config.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.add_imports) == 0
    assert f'{type(module_0._Config.remove_imports).__module__}.{type(module_0._Config.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.remove_imports) == 0
    assert module_0._Config.append_only is False
    assert module_0._Config.reverse_relative is False
    assert module_0._Config.force_single_line is False
    assert module_0._Config.single_line_exclusions == ()
    assert module_0._Config.default_section == 'THIRDPARTY'
    assert module_0._Config.balanced_wrapping is False
    assert module_0._Config.use_parentheses is False
    assert module_0._Config.order_by_type is True
    assert module_0._Config.atomic is False
    assert module_0._Config.lines_before_imports == -1
    assert module_0._Config.lines_after_imports == -1
    assert module_0._Config.lines_between_sections == 1
    assert module_0._Config.lines_between_types == 0
    assert module_0._Config.combine_as_imports is False
    assert module_0._Config.combine_star is False
    assert module_0._Config.include_trailing_comma is False
    assert module_0._Config.from_first is False
    assert module_0._Config.verbose is False
    assert module_0._Config.quiet is False
    assert module_0._Config.force_adds is False
    assert module_0._Config.force_alphabetical_sort_within_sections is False
    assert module_0._Config.force_alphabetical_sort is False
    assert module_0._Config.force_grid_wrap == 0
    assert module_0._Config.force_sort_within_sections is False
    assert module_0._Config.lexicographical is False
    assert module_0._Config.group_by_package is False
    assert module_0._Config.ignore_whitespace is False
    assert f'{type(module_0._Config.no_lines_before).__module__}.{type(module_0._Config.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.no_lines_before) == 0
    assert module_0._Config.no_inline_sort is False
    assert module_0._Config.ignore_comments is False
    assert module_0._Config.case_sensitive is False
    assert module_0._Config.sources == ()
    assert module_0._Config.virtual_env == ''
    assert module_0._Config.conda_env == ''
    assert module_0._Config.ensure_newline_before_comments is False
    assert module_0._Config.directory == ''
    assert module_0._Config.profile == ''
    assert module_0._Config.honor_noqa is False
    assert module_0._Config.src_paths == ()
    assert module_0._Config.remove_redundant_aliases is False
    assert module_0._Config.float_to_top is False
    assert module_0._Config.filter_files is False
    assert module_0._Config.formatter == ''
    assert module_0._Config.formatting_function is None
    assert module_0._Config.color_output is False
    assert f'{type(module_0._Config.treat_comments_as_code).__module__}.{type(module_0._Config.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.treat_comments_as_code) == 0
    assert module_0._Config.treat_all_comments_as_code is False
    assert f'{type(module_0._Config.supported_extensions).__module__}.{type(module_0._Config.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.supported_extensions) == 4
    assert f'{type(module_0._Config.blocked_extensions).__module__}.{type(module_0._Config.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.blocked_extensions) == 1
    assert f'{type(module_0._Config.constants).__module__}.{type(module_0._Config.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.constants) == 0
    assert f'{type(module_0._Config.classes).__module__}.{type(module_0._Config.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.classes) == 0
    assert f'{type(module_0._Config.variables).__module__}.{type(module_0._Config.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.variables) == 0
    assert module_0._Config.dedup_headings is False
    assert module_0._Config.only_sections is False
    assert module_0._Config.only_modified is False
    assert module_0._Config.combine_straight_imports is False
    assert module_0._Config.auto_identify_namespace_packages is True
    assert f'{type(module_0._Config.namespace_packages).__module__}.{type(module_0._Config.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.namespace_packages) == 0
    assert module_0._Config.follow_links is True
    assert module_0._Config.indented_import_headings is True
    assert module_0._Config.honor_case_in_force_sorted_sections is False
    assert module_0._Config.sort_relative_in_force_sorted_sections is False
    assert module_0._Config.overwrite_in_place is False
    assert module_0._Config.reverse_sort is False
    assert module_0._Config.star_first is False
    assert module_0._Config.format_error == '{error}: {message}'
    assert module_0._Config.format_success == '{success}: {message}'
    assert module_0._Config.sort_order == 'natural'
    assert module_0._Config.sort_reexports is False
    assert module_0._Config.split_on_trailing_comma is False

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'G|E\rp_)0Kp OH<='
    module_0.Config(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_3._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = 'rsMoe'
    module_0.Config(config=var_1)

def test_case_9():
    var_0 = 'G|E\rep_)0Kp OH<='
    var_1 = None
    var_2 = 'EX5\tfvkc&M3rU('
    var_3 = {var_0: var_1, var_2: var_1}
    with pytest.raises(module_4.UnsupportedSettings):
        module_0.Config(**var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_0.Config(config=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.settings.Config'
    assert var_1.py_version == 'py3'
    assert f'{type(var_1.force_to_top).__module__}.{type(var_1.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.force_to_top) == 0
    assert f'{type(var_1.skip).__module__}.{type(var_1.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.skip) == 19
    assert f'{type(var_1.extend_skip).__module__}.{type(var_1.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extend_skip) == 0
    assert f'{type(var_1.skip_glob).__module__}.{type(var_1.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.skip_glob) == 0
    assert f'{type(var_1.extend_skip_glob).__module__}.{type(var_1.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extend_skip_glob) == 0
    assert var_1.skip_gitignore is False
    assert var_1.line_length == 79
    assert var_1.wrap_length == 0
    assert var_1.line_ending == ''
    assert var_1.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_1.no_sections is False
    assert f'{type(var_1.known_future_library).__module__}.{type(var_1.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_future_library) == 1
    assert f'{type(var_1.known_third_party).__module__}.{type(var_1.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_third_party) == 0
    assert f'{type(var_1.known_first_party).__module__}.{type(var_1.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_first_party) == 0
    assert f'{type(var_1.known_local_folder).__module__}.{type(var_1.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_local_folder) == 0
    assert f'{type(var_1.known_standard_library).__module__}.{type(var_1.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_standard_library) == 234
    assert f'{type(var_1.extra_standard_library).__module__}.{type(var_1.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extra_standard_library) == 0
    assert var_1.known_other == {}
    assert var_1.multi_line_output == module_1.WrapModes.GRID
    assert var_1.forced_separate == ()
    assert var_1.indent == '    '
    assert var_1.comment_prefix == '  #'
    assert var_1.length_sort is False
    assert var_1.length_sort_straight is False
    assert f'{type(var_1.length_sort_sections).__module__}.{type(var_1.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.length_sort_sections) == 0
    assert f'{type(var_1.add_imports).__module__}.{type(var_1.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.add_imports) == 0
    assert f'{type(var_1.remove_imports).__module__}.{type(var_1.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.remove_imports) == 0
    assert var_1.append_only is False
    assert var_1.reverse_relative is False
    assert var_1.force_single_line is False
    assert var_1.single_line_exclusions == ()
    assert var_1.default_section == 'THIRDPARTY'
    assert var_1.import_headings == {}
    assert var_1.import_footers == {}
    assert var_1.balanced_wrapping is False
    assert var_1.use_parentheses is False
    assert var_1.order_by_type is True
    assert var_1.atomic is False
    assert var_1.lines_before_imports == -1
    assert var_1.lines_after_imports == -1
    assert var_1.lines_between_sections == 1
    assert var_1.lines_between_types == 0
    assert var_1.combine_as_imports is False
    assert var_1.combine_star is False
    assert var_1.include_trailing_comma is False
    assert var_1.from_first is False
    assert var_1.verbose is False
    assert var_1.quiet is False
    assert var_1.force_adds is False
    assert var_1.force_alphabetical_sort_within_sections is False
    assert var_1.force_alphabetical_sort is False
    assert var_1.force_grid_wrap == 0
    assert var_1.force_sort_within_sections is False
    assert var_1.lexicographical is False
    assert var_1.group_by_package is False
    assert var_1.ignore_whitespace is False
    assert f'{type(var_1.no_lines_before).__module__}.{type(var_1.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.no_lines_before) == 0
    assert var_1.no_inline_sort is False
    assert var_1.ignore_comments is False
    assert var_1.case_sensitive is False
    assert f'{type(var_1.sources).__module__}.{type(var_1.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_1.sources) == 1
    assert var_1.virtual_env == ''
    assert var_1.conda_env == ''
    assert var_1.ensure_newline_before_comments is False
    assert var_1.directory == '/workspace'
    assert var_1.profile == ''
    assert var_1.honor_noqa is False
    assert f'{type(var_1.src_paths).__module__}.{type(var_1.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_1.src_paths) == 2
    assert var_1.remove_redundant_aliases is False
    assert var_1.float_to_top is False
    assert var_1.filter_files is False
    assert var_1.formatter == ''
    assert var_1.formatting_function is None
    assert var_1.color_output is False
    assert f'{type(var_1.treat_comments_as_code).__module__}.{type(var_1.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.treat_comments_as_code) == 0
    assert var_1.treat_all_comments_as_code is False
    assert f'{type(var_1.supported_extensions).__module__}.{type(var_1.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.supported_extensions) == 4
    assert f'{type(var_1.blocked_extensions).__module__}.{type(var_1.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.blocked_extensions) == 1
    assert f'{type(var_1.constants).__module__}.{type(var_1.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.constants) == 0
    assert f'{type(var_1.classes).__module__}.{type(var_1.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.classes) == 0
    assert f'{type(var_1.variables).__module__}.{type(var_1.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.variables) == 0
    assert var_1.dedup_headings is False
    assert var_1.only_sections is False
    assert var_1.only_modified is False
    assert var_1.combine_straight_imports is False
    assert var_1.auto_identify_namespace_packages is True
    assert f'{type(var_1.namespace_packages).__module__}.{type(var_1.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.namespace_packages) == 0
    assert var_1.follow_links is True
    assert var_1.indented_import_headings is True
    assert var_1.honor_case_in_force_sorted_sections is False
    assert var_1.sort_relative_in_force_sorted_sections is False
    assert var_1.overwrite_in_place is False
    assert var_1.reverse_sort is False
    assert var_1.star_first is False
    assert var_1.git_ls_files == {}
    assert var_1.format_error == '{error}: {message}'
    assert var_1.format_success == '{success}: {message}'
    assert var_1.sort_order == 'natural'
    assert var_1.sort_reexports is False
    assert var_1.split_on_trailing_comma is False
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_0.Config.known_patterns).__module__}.{type(module_0.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments).__module__}.{type(module_0.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments_end).__module__}.{type(module_0.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skips).__module__}.{type(module_0.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skip_globs).__module__}.{type(module_0.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.sorting_function).__module__}.{type(module_0.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_2 = module_3._EnumDict()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'enum._EnumDict'
    assert len(var_2) == 0
    var_3 = 'UtM'
    var_4 = var_1.__hash__()
    var_5 = module_0.entry_points(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'importlib.metadata.EntryPoints'
    assert len(var_5) == 0
    assert f'{type(module_2.EntryPoints.names).__module__}.{type(module_2.EntryPoints.names).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.EntryPoints.groups).__module__}.{type(module_2.EntryPoints.groups).__qualname__}' == 'builtins.property'
    module_0.Config(**var_5)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.Config(config=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.settings.Config'
    assert var_1.py_version == 'py3'
    assert f'{type(var_1.force_to_top).__module__}.{type(var_1.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.force_to_top) == 0
    assert f'{type(var_1.skip).__module__}.{type(var_1.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.skip) == 19
    assert f'{type(var_1.extend_skip).__module__}.{type(var_1.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extend_skip) == 0
    assert f'{type(var_1.skip_glob).__module__}.{type(var_1.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.skip_glob) == 0
    assert f'{type(var_1.extend_skip_glob).__module__}.{type(var_1.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extend_skip_glob) == 0
    assert var_1.skip_gitignore is False
    assert var_1.line_length == 79
    assert var_1.wrap_length == 0
    assert var_1.line_ending == ''
    assert var_1.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_1.no_sections is False
    assert f'{type(var_1.known_future_library).__module__}.{type(var_1.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_future_library) == 1
    assert f'{type(var_1.known_third_party).__module__}.{type(var_1.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_third_party) == 0
    assert f'{type(var_1.known_first_party).__module__}.{type(var_1.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_first_party) == 0
    assert f'{type(var_1.known_local_folder).__module__}.{type(var_1.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_local_folder) == 0
    assert f'{type(var_1.known_standard_library).__module__}.{type(var_1.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_standard_library) == 234
    assert f'{type(var_1.extra_standard_library).__module__}.{type(var_1.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extra_standard_library) == 0
    assert var_1.known_other == {}
    assert var_1.multi_line_output == module_1.WrapModes.GRID
    assert var_1.forced_separate == ()
    assert var_1.indent == '    '
    assert var_1.comment_prefix == '  #'
    assert var_1.length_sort is False
    assert var_1.length_sort_straight is False
    assert f'{type(var_1.length_sort_sections).__module__}.{type(var_1.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.length_sort_sections) == 0
    assert f'{type(var_1.add_imports).__module__}.{type(var_1.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.add_imports) == 0
    assert f'{type(var_1.remove_imports).__module__}.{type(var_1.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.remove_imports) == 0
    assert var_1.append_only is False
    assert var_1.reverse_relative is False
    assert var_1.force_single_line is False
    assert var_1.single_line_exclusions == ()
    assert var_1.default_section == 'THIRDPARTY'
    assert var_1.import_headings == {}
    assert var_1.import_footers == {}
    assert var_1.balanced_wrapping is False
    assert var_1.use_parentheses is False
    assert var_1.order_by_type is True
    assert var_1.atomic is False
    assert var_1.lines_before_imports == -1
    assert var_1.lines_after_imports == -1
    assert var_1.lines_between_sections == 1
    assert var_1.lines_between_types == 0
    assert var_1.combine_as_imports is False
    assert var_1.combine_star is False
    assert var_1.include_trailing_comma is False
    assert var_1.from_first is False
    assert var_1.verbose is False
    assert var_1.quiet is False
    assert var_1.force_adds is False
    assert var_1.force_alphabetical_sort_within_sections is False
    assert var_1.force_alphabetical_sort is False
    assert var_1.force_grid_wrap == 0
    assert var_1.force_sort_within_sections is False
    assert var_1.lexicographical is False
    assert var_1.group_by_package is False
    assert var_1.ignore_whitespace is False
    assert f'{type(var_1.no_lines_before).__module__}.{type(var_1.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.no_lines_before) == 0
    assert var_1.no_inline_sort is False
    assert var_1.ignore_comments is False
    assert var_1.case_sensitive is False
    assert f'{type(var_1.sources).__module__}.{type(var_1.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_1.sources) == 1
    assert var_1.virtual_env == ''
    assert var_1.conda_env == ''
    assert var_1.ensure_newline_before_comments is False
    assert var_1.directory == '/workspace'
    assert var_1.profile == ''
    assert var_1.honor_noqa is False
    assert f'{type(var_1.src_paths).__module__}.{type(var_1.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_1.src_paths) == 2
    assert var_1.remove_redundant_aliases is False
    assert var_1.float_to_top is False
    assert var_1.filter_files is False
    assert var_1.formatter == ''
    assert var_1.formatting_function is None
    assert var_1.color_output is False
    assert f'{type(var_1.treat_comments_as_code).__module__}.{type(var_1.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.treat_comments_as_code) == 0
    assert var_1.treat_all_comments_as_code is False
    assert f'{type(var_1.supported_extensions).__module__}.{type(var_1.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.supported_extensions) == 4
    assert f'{type(var_1.blocked_extensions).__module__}.{type(var_1.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.blocked_extensions) == 1
    assert f'{type(var_1.constants).__module__}.{type(var_1.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.constants) == 0
    assert f'{type(var_1.classes).__module__}.{type(var_1.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.classes) == 0
    assert f'{type(var_1.variables).__module__}.{type(var_1.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.variables) == 0
    assert var_1.dedup_headings is False
    assert var_1.only_sections is False
    assert var_1.only_modified is False
    assert var_1.combine_straight_imports is False
    assert var_1.auto_identify_namespace_packages is True
    assert f'{type(var_1.namespace_packages).__module__}.{type(var_1.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.namespace_packages) == 0
    assert var_1.follow_links is True
    assert var_1.indented_import_headings is True
    assert var_1.honor_case_in_force_sorted_sections is False
    assert var_1.sort_relative_in_force_sorted_sections is False
    assert var_1.overwrite_in_place is False
    assert var_1.reverse_sort is False
    assert var_1.star_first is False
    assert var_1.git_ls_files == {}
    assert var_1.format_error == '{error}: {message}'
    assert var_1.format_success == '{success}: {message}'
    assert var_1.sort_order == 'natural'
    assert var_1.sort_reexports is False
    assert var_1.split_on_trailing_comma is False
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_0.Config.known_patterns).__module__}.{type(module_0.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments).__module__}.{type(module_0.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments_end).__module__}.{type(module_0.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skips).__module__}.{type(module_0.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skip_globs).__module__}.{type(module_0.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.sorting_function).__module__}.{type(module_0.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_2 = 'zu~Znp!;~;FuI'
    var_3 = module_0.find_all_configs(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_3.root).__module__}.{type(var_3.root).__qualname__}' == 'isort.utils.TrieNode'
    var_1.is_skipped(var_0)

def test_case_12():
    var_0 = '//'
    var_1 = module_0.find_all_configs(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    var_2 = module_0.Config(settings_path=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.settings.Config'
    assert var_2.py_version == 'py3'
    assert f'{type(var_2.force_to_top).__module__}.{type(var_2.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.force_to_top) == 0
    assert f'{type(var_2.skip).__module__}.{type(var_2.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.skip) == 19
    assert f'{type(var_2.extend_skip).__module__}.{type(var_2.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.extend_skip) == 0
    assert f'{type(var_2.skip_glob).__module__}.{type(var_2.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.skip_glob) == 0
    assert f'{type(var_2.extend_skip_glob).__module__}.{type(var_2.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.extend_skip_glob) == 0
    assert var_2.skip_gitignore is False
    assert var_2.line_length == 79
    assert var_2.wrap_length == 0
    assert var_2.line_ending == ''
    assert var_2.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_2.no_sections is False
    assert f'{type(var_2.known_future_library).__module__}.{type(var_2.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_future_library) == 1
    assert f'{type(var_2.known_third_party).__module__}.{type(var_2.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_third_party) == 0
    assert f'{type(var_2.known_first_party).__module__}.{type(var_2.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_first_party) == 0
    assert f'{type(var_2.known_local_folder).__module__}.{type(var_2.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_local_folder) == 0
    assert f'{type(var_2.known_standard_library).__module__}.{type(var_2.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_standard_library) == 234
    assert f'{type(var_2.extra_standard_library).__module__}.{type(var_2.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.extra_standard_library) == 0
    assert var_2.known_other == {}
    assert var_2.multi_line_output == module_1.WrapModes.GRID
    assert var_2.forced_separate == ()
    assert var_2.indent == '    '
    assert var_2.comment_prefix == '  #'
    assert var_2.length_sort is False
    assert var_2.length_sort_straight is False
    assert f'{type(var_2.length_sort_sections).__module__}.{type(var_2.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.length_sort_sections) == 0
    assert f'{type(var_2.add_imports).__module__}.{type(var_2.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.add_imports) == 0
    assert f'{type(var_2.remove_imports).__module__}.{type(var_2.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.remove_imports) == 0
    assert var_2.append_only is False
    assert var_2.reverse_relative is False
    assert var_2.force_single_line is False
    assert var_2.single_line_exclusions == ()
    assert var_2.default_section == 'THIRDPARTY'
    assert var_2.import_headings == {}
    assert var_2.import_footers == {}
    assert var_2.balanced_wrapping is False
    assert var_2.use_parentheses is False
    assert var_2.order_by_type is True
    assert var_2.atomic is False
    assert var_2.lines_before_imports == -1
    assert var_2.lines_after_imports == -1
    assert var_2.lines_between_sections == 1
    assert var_2.lines_between_types == 0
    assert var_2.combine_as_imports is False
    assert var_2.combine_star is False
    assert var_2.include_trailing_comma is False
    assert var_2.from_first is False
    assert var_2.verbose is False
    assert var_2.quiet is False
    assert var_2.force_adds is False
    assert var_2.force_alphabetical_sort_within_sections is False
    assert var_2.force_alphabetical_sort is False
    assert var_2.force_grid_wrap == 0
    assert var_2.force_sort_within_sections is False
    assert var_2.lexicographical is False
    assert var_2.group_by_package is False
    assert var_2.ignore_whitespace is False
    assert f'{type(var_2.no_lines_before).__module__}.{type(var_2.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.no_lines_before) == 0
    assert var_2.no_inline_sort is False
    assert var_2.ignore_comments is False
    assert var_2.case_sensitive is False
    assert f'{type(var_2.sources).__module__}.{type(var_2.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_2.sources) == 1
    assert var_2.virtual_env == ''
    assert var_2.conda_env == ''
    assert var_2.ensure_newline_before_comments is False
    assert var_2.directory == '/workspace'
    assert var_2.profile == ''
    assert var_2.honor_noqa is False
    assert f'{type(var_2.src_paths).__module__}.{type(var_2.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_2.src_paths) == 2
    assert var_2.remove_redundant_aliases is False
    assert var_2.float_to_top is False
    assert var_2.filter_files is False
    assert var_2.formatter == ''
    assert var_2.formatting_function is None
    assert var_2.color_output is False
    assert f'{type(var_2.treat_comments_as_code).__module__}.{type(var_2.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.treat_comments_as_code) == 0
    assert var_2.treat_all_comments_as_code is False
    assert f'{type(var_2.supported_extensions).__module__}.{type(var_2.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.supported_extensions) == 4
    assert f'{type(var_2.blocked_extensions).__module__}.{type(var_2.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.blocked_extensions) == 1
    assert f'{type(var_2.constants).__module__}.{type(var_2.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.constants) == 0
    assert f'{type(var_2.classes).__module__}.{type(var_2.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.classes) == 0
    assert f'{type(var_2.variables).__module__}.{type(var_2.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.variables) == 0
    assert var_2.dedup_headings is False
    assert var_2.only_sections is False
    assert var_2.only_modified is False
    assert var_2.combine_straight_imports is False
    assert var_2.auto_identify_namespace_packages is True
    assert f'{type(var_2.namespace_packages).__module__}.{type(var_2.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.namespace_packages) == 0
    assert var_2.follow_links is True
    assert var_2.indented_import_headings is True
    assert var_2.honor_case_in_force_sorted_sections is False
    assert var_2.sort_relative_in_force_sorted_sections is False
    assert var_2.overwrite_in_place is False
    assert var_2.reverse_sort is False
    assert var_2.star_first is False
    assert var_2.git_ls_files == {}
    assert var_2.format_error == '{error}: {message}'
    assert var_2.format_success == '{success}: {message}'
    assert var_2.sort_order == 'natural'
    assert var_2.sort_reexports is False
    assert var_2.split_on_trailing_comma is False
    assert f'{type(module_0.Config.known_patterns).__module__}.{type(module_0.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments).__module__}.{type(module_0.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments_end).__module__}.{type(module_0.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skips).__module__}.{type(module_0.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skip_globs).__module__}.{type(module_0.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.sorting_function).__module__}.{type(module_0.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_3 = var_2.is_supported_filetype(var_0)
    assert var_3 is False

def test_case_13():
    var_0 = module_0.Config()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.settings.Config'
    assert var_0.py_version == 'py3'
    assert f'{type(var_0.force_to_top).__module__}.{type(var_0.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.force_to_top) == 0
    assert f'{type(var_0.skip).__module__}.{type(var_0.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.skip) == 19
    assert f'{type(var_0.extend_skip).__module__}.{type(var_0.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.extend_skip) == 0
    assert f'{type(var_0.skip_glob).__module__}.{type(var_0.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.skip_glob) == 0
    assert f'{type(var_0.extend_skip_glob).__module__}.{type(var_0.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.extend_skip_glob) == 0
    assert var_0.skip_gitignore is False
    assert var_0.line_length == 79
    assert var_0.wrap_length == 0
    assert var_0.line_ending == ''
    assert var_0.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_0.no_sections is False
    assert f'{type(var_0.known_future_library).__module__}.{type(var_0.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_future_library) == 1
    assert f'{type(var_0.known_third_party).__module__}.{type(var_0.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_third_party) == 0
    assert f'{type(var_0.known_first_party).__module__}.{type(var_0.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_first_party) == 0
    assert f'{type(var_0.known_local_folder).__module__}.{type(var_0.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_local_folder) == 0
    assert f'{type(var_0.known_standard_library).__module__}.{type(var_0.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_standard_library) == 234
    assert f'{type(var_0.extra_standard_library).__module__}.{type(var_0.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.extra_standard_library) == 0
    assert var_0.known_other == {}
    assert var_0.multi_line_output == module_1.WrapModes.GRID
    assert var_0.forced_separate == ()
    assert var_0.indent == '    '
    assert var_0.comment_prefix == '  #'
    assert var_0.length_sort is False
    assert var_0.length_sort_straight is False
    assert f'{type(var_0.length_sort_sections).__module__}.{type(var_0.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.length_sort_sections) == 0
    assert f'{type(var_0.add_imports).__module__}.{type(var_0.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.add_imports) == 0
    assert f'{type(var_0.remove_imports).__module__}.{type(var_0.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.remove_imports) == 0
    assert var_0.append_only is False
    assert var_0.reverse_relative is False
    assert var_0.force_single_line is False
    assert var_0.single_line_exclusions == ()
    assert var_0.default_section == 'THIRDPARTY'
    assert var_0.import_headings == {}
    assert var_0.import_footers == {}
    assert var_0.balanced_wrapping is False
    assert var_0.use_parentheses is False
    assert var_0.order_by_type is True
    assert var_0.atomic is False
    assert var_0.lines_before_imports == -1
    assert var_0.lines_after_imports == -1
    assert var_0.lines_between_sections == 1
    assert var_0.lines_between_types == 0
    assert var_0.combine_as_imports is False
    assert var_0.combine_star is False
    assert var_0.include_trailing_comma is False
    assert var_0.from_first is False
    assert var_0.verbose is False
    assert var_0.quiet is False
    assert var_0.force_adds is False
    assert var_0.force_alphabetical_sort_within_sections is False
    assert var_0.force_alphabetical_sort is False
    assert var_0.force_grid_wrap == 0
    assert var_0.force_sort_within_sections is False
    assert var_0.lexicographical is False
    assert var_0.group_by_package is False
    assert var_0.ignore_whitespace is False
    assert f'{type(var_0.no_lines_before).__module__}.{type(var_0.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.no_lines_before) == 0
    assert var_0.no_inline_sort is False
    assert var_0.ignore_comments is False
    assert var_0.case_sensitive is False
    assert f'{type(var_0.sources).__module__}.{type(var_0.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_0.sources) == 1
    assert var_0.virtual_env == ''
    assert var_0.conda_env == ''
    assert var_0.ensure_newline_before_comments is False
    assert var_0.directory == '/workspace'
    assert var_0.profile == ''
    assert var_0.honor_noqa is False
    assert f'{type(var_0.src_paths).__module__}.{type(var_0.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_0.src_paths) == 2
    assert var_0.remove_redundant_aliases is False
    assert var_0.float_to_top is False
    assert var_0.filter_files is False
    assert var_0.formatter == ''
    assert var_0.formatting_function is None
    assert var_0.color_output is False
    assert f'{type(var_0.treat_comments_as_code).__module__}.{type(var_0.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.treat_comments_as_code) == 0
    assert var_0.treat_all_comments_as_code is False
    assert f'{type(var_0.supported_extensions).__module__}.{type(var_0.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.supported_extensions) == 4
    assert f'{type(var_0.blocked_extensions).__module__}.{type(var_0.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.blocked_extensions) == 1
    assert f'{type(var_0.constants).__module__}.{type(var_0.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.constants) == 0
    assert f'{type(var_0.classes).__module__}.{type(var_0.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.classes) == 0
    assert f'{type(var_0.variables).__module__}.{type(var_0.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.variables) == 0
    assert var_0.dedup_headings is False
    assert var_0.only_sections is False
    assert var_0.only_modified is False
    assert var_0.combine_straight_imports is False
    assert var_0.auto_identify_namespace_packages is True
    assert f'{type(var_0.namespace_packages).__module__}.{type(var_0.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.namespace_packages) == 0
    assert var_0.follow_links is True
    assert var_0.indented_import_headings is True
    assert var_0.honor_case_in_force_sorted_sections is False
    assert var_0.sort_relative_in_force_sorted_sections is False
    assert var_0.overwrite_in_place is False
    assert var_0.reverse_sort is False
    assert var_0.star_first is False
    assert var_0.git_ls_files == {}
    assert var_0.format_error == '{error}: {message}'
    assert var_0.format_success == '{success}: {message}'
    assert var_0.sort_order == 'natural'
    assert var_0.sort_reexports is False
    assert var_0.split_on_trailing_comma is False
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_0.Config.known_patterns).__module__}.{type(module_0.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments).__module__}.{type(module_0.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments_end).__module__}.{type(module_0.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skips).__module__}.{type(module_0.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skip_globs).__module__}.{type(module_0.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.sorting_function).__module__}.{type(module_0.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_1 = 'k5wWJ=@A"r6rRZ;AvOZ'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = True
    var_2 = [var_1, var_0, var_0]
    var_3 = (var_1, var_2)
    var_4 = False
    var_5 = 'F2KYN/\nc\rs.M['
    var_6 = (var_5, var_1)
    var_7 = (var_4, var_6, var_3)
    var_8 = False
    var_9 = True
    var_10 = True
    var_11 = '\';oAv,{e\n<;v$"nL(\x0b'
    var_12 = True
    var_13 = True
    var_14 = module_0._Config(skip=var_0, skip_gitignore=var_1, wrap_length=var_1, sections=var_0, extra_standard_library=var_3, remove_imports=var_7, append_only=var_8, force_single_line=var_8, balanced_wrapping=var_4, use_parentheses=var_4, atomic=var_0, lines_before_imports=var_8, lines_between_types=var_0, verbose=var_9, force_alphabetical_sort=var_1, force_sort_within_sections=var_10, no_inline_sort=var_1, directory=var_11, remove_redundant_aliases=var_0, filter_files=var_0, color_output=var_8, constants=var_5, dedup_headings=var_10, indented_import_headings=var_0, star_first=var_11, sort_order=var_11, sort_reexports=var_12, split_on_trailing_comma=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'isort.settings._Config'
    assert var_14.py_version == 'py3'
    assert f'{type(var_14.force_to_top).__module__}.{type(var_14.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.force_to_top) == 0
    assert var_14.skip is None
    assert f'{type(var_14.extend_skip).__module__}.{type(var_14.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extend_skip) == 0
    assert f'{type(var_14.skip_glob).__module__}.{type(var_14.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.skip_glob) == 0
    assert f'{type(var_14.extend_skip_glob).__module__}.{type(var_14.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extend_skip_glob) == 0
    assert var_14.skip_gitignore is True
    assert var_14.line_length == 79
    assert var_14.wrap_length is True
    assert var_14.line_ending == ''
    assert var_14.sections is None
    assert var_14.no_sections is True
    assert f'{type(var_14.known_future_library).__module__}.{type(var_14.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_future_library) == 1
    assert f'{type(var_14.known_third_party).__module__}.{type(var_14.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_third_party) == 0
    assert f'{type(var_14.known_first_party).__module__}.{type(var_14.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_first_party) == 0
    assert f'{type(var_14.known_local_folder).__module__}.{type(var_14.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_local_folder) == 0
    assert f'{type(var_14.known_standard_library).__module__}.{type(var_14.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_standard_library) == 234
    assert var_14.extra_standard_library == (True, [True, None, None])
    assert var_14.known_other == {}
    assert var_14.multi_line_output == module_1.WrapModes.GRID
    assert var_14.forced_separate == ()
    assert var_14.indent == '    '
    assert var_14.comment_prefix == '  #'
    assert var_14.length_sort is False
    assert var_14.length_sort_straight is False
    assert f'{type(var_14.length_sort_sections).__module__}.{type(var_14.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.length_sort_sections) == 0
    assert f'{type(var_14.add_imports).__module__}.{type(var_14.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.add_imports) == 0
    assert var_14.remove_imports == (False, ('F2KYN/\nc\rs.M[', True), (True, [True, None, None]))
    assert var_14.append_only is False
    assert var_14.reverse_relative is False
    assert var_14.force_single_line is False
    assert var_14.single_line_exclusions == ()
    assert var_14.default_section == 'THIRDPARTY'
    assert var_14.import_headings == {}
    assert var_14.import_footers == {}
    assert var_14.balanced_wrapping is False
    assert var_14.use_parentheses is False
    assert var_14.order_by_type is True
    assert var_14.atomic is None
    assert var_14.lines_before_imports is False
    assert var_14.lines_after_imports == -1
    assert var_14.lines_between_sections == 1
    assert var_14.lines_between_types == 1
    assert var_14.combine_as_imports is False
    assert var_14.combine_star is False
    assert var_14.include_trailing_comma is False
    assert var_14.from_first is True
    assert var_14.verbose is True
    assert var_14.quiet is False
    assert var_14.force_adds is False
    assert var_14.force_alphabetical_sort_within_sections is True
    assert var_14.force_alphabetical_sort is True
    assert var_14.force_grid_wrap == 0
    assert var_14.force_sort_within_sections is True
    assert var_14.lexicographical is False
    assert var_14.group_by_package is False
    assert var_14.ignore_whitespace is False
    assert f'{type(var_14.no_lines_before).__module__}.{type(var_14.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.no_lines_before) == 0
    assert var_14.no_inline_sort is True
    assert var_14.ignore_comments is False
    assert var_14.case_sensitive is False
    assert var_14.sources == ()
    assert var_14.virtual_env == ''
    assert var_14.conda_env == ''
    assert var_14.ensure_newline_before_comments is False
    assert var_14.directory == '\';oAv,{e\n<;v$"nL(\x0b'
    assert var_14.profile == ''
    assert var_14.honor_noqa is False
    assert var_14.src_paths == ()
    assert var_14.remove_redundant_aliases is None
    assert var_14.float_to_top is False
    assert var_14.filter_files is None
    assert var_14.formatter == ''
    assert var_14.formatting_function is None
    assert var_14.color_output is False
    assert f'{type(var_14.treat_comments_as_code).__module__}.{type(var_14.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.treat_comments_as_code) == 0
    assert var_14.treat_all_comments_as_code is False
    assert f'{type(var_14.supported_extensions).__module__}.{type(var_14.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.supported_extensions) == 4
    assert f'{type(var_14.blocked_extensions).__module__}.{type(var_14.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.blocked_extensions) == 1
    assert var_14.constants == 'F2KYN/\nc\rs.M['
    assert f'{type(var_14.classes).__module__}.{type(var_14.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.classes) == 0
    assert f'{type(var_14.variables).__module__}.{type(var_14.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.variables) == 0
    assert var_14.dedup_headings is True
    assert var_14.only_sections is False
    assert var_14.only_modified is False
    assert var_14.combine_straight_imports is False
    assert var_14.auto_identify_namespace_packages is True
    assert f'{type(var_14.namespace_packages).__module__}.{type(var_14.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.namespace_packages) == 0
    assert var_14.follow_links is True
    assert var_14.indented_import_headings is None
    assert var_14.honor_case_in_force_sorted_sections is False
    assert var_14.sort_relative_in_force_sorted_sections is False
    assert var_14.overwrite_in_place is False
    assert var_14.reverse_sort is False
    assert var_14.star_first == '\';oAv,{e\n<;v$"nL(\x0b'
    assert var_14.git_ls_files == {}
    assert var_14.format_error == '{error}: {message}'
    assert var_14.format_success == '{success}: {message}'
    assert var_14.sort_order == '\';oAv,{e\n<;v$"nL(\x0b'
    assert var_14.sort_reexports is True
    assert var_14.split_on_trailing_comma is True
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert module_0._Config.py_version == '3'
    assert f'{type(module_0._Config.force_to_top).__module__}.{type(module_0._Config.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.force_to_top) == 0
    assert f'{type(module_0._Config.skip).__module__}.{type(module_0._Config.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.skip) == 19
    assert f'{type(module_0._Config.extend_skip).__module__}.{type(module_0._Config.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.extend_skip) == 0
    assert f'{type(module_0._Config.skip_glob).__module__}.{type(module_0._Config.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.skip_glob) == 0
    assert f'{type(module_0._Config.extend_skip_glob).__module__}.{type(module_0._Config.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.extend_skip_glob) == 0
    assert module_0._Config.skip_gitignore is False
    assert module_0._Config.line_length == 79
    assert module_0._Config.wrap_length == 0
    assert module_0._Config.line_ending == ''
    assert module_0._Config.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0._Config.no_sections is False
    assert f'{type(module_0._Config.known_future_library).__module__}.{type(module_0._Config.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_future_library) == 1
    assert f'{type(module_0._Config.known_third_party).__module__}.{type(module_0._Config.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_third_party) == 0
    assert f'{type(module_0._Config.known_first_party).__module__}.{type(module_0._Config.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_first_party) == 0
    assert f'{type(module_0._Config.known_local_folder).__module__}.{type(module_0._Config.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_local_folder) == 0
    assert f'{type(module_0._Config.known_standard_library).__module__}.{type(module_0._Config.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.known_standard_library) == 0
    assert f'{type(module_0._Config.extra_standard_library).__module__}.{type(module_0._Config.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.extra_standard_library) == 0
    assert module_0._Config.multi_line_output == module_1.WrapModes.GRID
    assert module_0._Config.forced_separate == ()
    assert module_0._Config.indent == '    '
    assert module_0._Config.comment_prefix == '  #'
    assert module_0._Config.length_sort is False
    assert module_0._Config.length_sort_straight is False
    assert f'{type(module_0._Config.length_sort_sections).__module__}.{type(module_0._Config.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.length_sort_sections) == 0
    assert f'{type(module_0._Config.add_imports).__module__}.{type(module_0._Config.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.add_imports) == 0
    assert f'{type(module_0._Config.remove_imports).__module__}.{type(module_0._Config.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.remove_imports) == 0
    assert module_0._Config.append_only is False
    assert module_0._Config.reverse_relative is False
    assert module_0._Config.force_single_line is False
    assert module_0._Config.single_line_exclusions == ()
    assert module_0._Config.default_section == 'THIRDPARTY'
    assert module_0._Config.balanced_wrapping is False
    assert module_0._Config.use_parentheses is False
    assert module_0._Config.order_by_type is True
    assert module_0._Config.atomic is False
    assert module_0._Config.lines_before_imports == -1
    assert module_0._Config.lines_after_imports == -1
    assert module_0._Config.lines_between_sections == 1
    assert module_0._Config.lines_between_types == 0
    assert module_0._Config.combine_as_imports is False
    assert module_0._Config.combine_star is False
    assert module_0._Config.include_trailing_comma is False
    assert module_0._Config.from_first is False
    assert module_0._Config.verbose is False
    assert module_0._Config.quiet is False
    assert module_0._Config.force_adds is False
    assert module_0._Config.force_alphabetical_sort_within_sections is False
    assert module_0._Config.force_alphabetical_sort is False
    assert module_0._Config.force_grid_wrap == 0
    assert module_0._Config.force_sort_within_sections is False
    assert module_0._Config.lexicographical is False
    assert module_0._Config.group_by_package is False
    assert module_0._Config.ignore_whitespace is False
    assert f'{type(module_0._Config.no_lines_before).__module__}.{type(module_0._Config.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.no_lines_before) == 0
    assert module_0._Config.no_inline_sort is False
    assert module_0._Config.ignore_comments is False
    assert module_0._Config.case_sensitive is False
    assert module_0._Config.sources == ()
    assert module_0._Config.virtual_env == ''
    assert module_0._Config.conda_env == ''
    assert module_0._Config.ensure_newline_before_comments is False
    assert module_0._Config.directory == ''
    assert module_0._Config.profile == ''
    assert module_0._Config.honor_noqa is False
    assert module_0._Config.src_paths == ()
    assert module_0._Config.remove_redundant_aliases is False
    assert module_0._Config.float_to_top is False
    assert module_0._Config.filter_files is False
    assert module_0._Config.formatter == ''
    assert module_0._Config.formatting_function is None
    assert module_0._Config.color_output is False
    assert f'{type(module_0._Config.treat_comments_as_code).__module__}.{type(module_0._Config.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.treat_comments_as_code) == 0
    assert module_0._Config.treat_all_comments_as_code is False
    assert f'{type(module_0._Config.supported_extensions).__module__}.{type(module_0._Config.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.supported_extensions) == 4
    assert f'{type(module_0._Config.blocked_extensions).__module__}.{type(module_0._Config.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.blocked_extensions) == 1
    assert f'{type(module_0._Config.constants).__module__}.{type(module_0._Config.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.constants) == 0
    assert f'{type(module_0._Config.classes).__module__}.{type(module_0._Config.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.classes) == 0
    assert f'{type(module_0._Config.variables).__module__}.{type(module_0._Config.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.variables) == 0
    assert module_0._Config.dedup_headings is False
    assert module_0._Config.only_sections is False
    assert module_0._Config.only_modified is False
    assert module_0._Config.combine_straight_imports is False
    assert module_0._Config.auto_identify_namespace_packages is True
    assert f'{type(module_0._Config.namespace_packages).__module__}.{type(module_0._Config.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0._Config.namespace_packages) == 0
    assert module_0._Config.follow_links is True
    assert module_0._Config.indented_import_headings is True
    assert module_0._Config.honor_case_in_force_sorted_sections is False
    assert module_0._Config.sort_relative_in_force_sorted_sections is False
    assert module_0._Config.overwrite_in_place is False
    assert module_0._Config.reverse_sort is False
    assert module_0._Config.star_first is False
    assert module_0._Config.format_error == '{error}: {message}'
    assert module_0._Config.format_success == '{success}: {message}'
    assert module_0._Config.sort_order == 'natural'
    assert module_0._Config.sort_reexports is False
    assert module_0._Config.split_on_trailing_comma is False
    module_3.unique(var_1)

def test_case_15():
    var_0 = module_0.Config()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.settings.Config'
    assert var_0.py_version == 'py3'
    assert f'{type(var_0.force_to_top).__module__}.{type(var_0.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.force_to_top) == 0
    assert f'{type(var_0.skip).__module__}.{type(var_0.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.skip) == 19
    assert f'{type(var_0.extend_skip).__module__}.{type(var_0.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.extend_skip) == 0
    assert f'{type(var_0.skip_glob).__module__}.{type(var_0.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.skip_glob) == 0
    assert f'{type(var_0.extend_skip_glob).__module__}.{type(var_0.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.extend_skip_glob) == 0
    assert var_0.skip_gitignore is False
    assert var_0.line_length == 79
    assert var_0.wrap_length == 0
    assert var_0.line_ending == ''
    assert var_0.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_0.no_sections is False
    assert f'{type(var_0.known_future_library).__module__}.{type(var_0.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_future_library) == 1
    assert f'{type(var_0.known_third_party).__module__}.{type(var_0.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_third_party) == 0
    assert f'{type(var_0.known_first_party).__module__}.{type(var_0.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_first_party) == 0
    assert f'{type(var_0.known_local_folder).__module__}.{type(var_0.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_local_folder) == 0
    assert f'{type(var_0.known_standard_library).__module__}.{type(var_0.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_standard_library) == 234
    assert f'{type(var_0.extra_standard_library).__module__}.{type(var_0.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.extra_standard_library) == 0
    assert var_0.known_other == {}
    assert var_0.multi_line_output == module_1.WrapModes.GRID
    assert var_0.forced_separate == ()
    assert var_0.indent == '    '
    assert var_0.comment_prefix == '  #'
    assert var_0.length_sort is False
    assert var_0.length_sort_straight is False
    assert f'{type(var_0.length_sort_sections).__module__}.{type(var_0.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.length_sort_sections) == 0
    assert f'{type(var_0.add_imports).__module__}.{type(var_0.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.add_imports) == 0
    assert f'{type(var_0.remove_imports).__module__}.{type(var_0.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.remove_imports) == 0
    assert var_0.append_only is False
    assert var_0.reverse_relative is False
    assert var_0.force_single_line is False
    assert var_0.single_line_exclusions == ()
    assert var_0.default_section == 'THIRDPARTY'
    assert var_0.import_headings == {}
    assert var_0.import_footers == {}
    assert var_0.balanced_wrapping is False
    assert var_0.use_parentheses is False
    assert var_0.order_by_type is True
    assert var_0.atomic is False
    assert var_0.lines_before_imports == -1
    assert var_0.lines_after_imports == -1
    assert var_0.lines_between_sections == 1
    assert var_0.lines_between_types == 0
    assert var_0.combine_as_imports is False
    assert var_0.combine_star is False
    assert var_0.include_trailing_comma is False
    assert var_0.from_first is False
    assert var_0.verbose is False
    assert var_0.quiet is False
    assert var_0.force_adds is False
    assert var_0.force_alphabetical_sort_within_sections is False
    assert var_0.force_alphabetical_sort is False
    assert var_0.force_grid_wrap == 0
    assert var_0.force_sort_within_sections is False
    assert var_0.lexicographical is False
    assert var_0.group_by_package is False
    assert var_0.ignore_whitespace is False
    assert f'{type(var_0.no_lines_before).__module__}.{type(var_0.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.no_lines_before) == 0
    assert var_0.no_inline_sort is False
    assert var_0.ignore_comments is False
    assert var_0.case_sensitive is False
    assert f'{type(var_0.sources).__module__}.{type(var_0.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_0.sources) == 1
    assert var_0.virtual_env == ''
    assert var_0.conda_env == ''
    assert var_0.ensure_newline_before_comments is False
    assert var_0.directory == '/workspace'
    assert var_0.profile == ''
    assert var_0.honor_noqa is False
    assert f'{type(var_0.src_paths).__module__}.{type(var_0.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_0.src_paths) == 2
    assert var_0.remove_redundant_aliases is False
    assert var_0.float_to_top is False
    assert var_0.filter_files is False
    assert var_0.formatter == ''
    assert var_0.formatting_function is None
    assert var_0.color_output is False
    assert f'{type(var_0.treat_comments_as_code).__module__}.{type(var_0.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.treat_comments_as_code) == 0
    assert var_0.treat_all_comments_as_code is False
    assert f'{type(var_0.supported_extensions).__module__}.{type(var_0.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.supported_extensions) == 4
    assert f'{type(var_0.blocked_extensions).__module__}.{type(var_0.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.blocked_extensions) == 1
    assert f'{type(var_0.constants).__module__}.{type(var_0.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.constants) == 0
    assert f'{type(var_0.classes).__module__}.{type(var_0.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.classes) == 0
    assert f'{type(var_0.variables).__module__}.{type(var_0.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.variables) == 0
    assert var_0.dedup_headings is False
    assert var_0.only_sections is False
    assert var_0.only_modified is False
    assert var_0.combine_straight_imports is False
    assert var_0.auto_identify_namespace_packages is True
    assert f'{type(var_0.namespace_packages).__module__}.{type(var_0.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.namespace_packages) == 0
    assert var_0.follow_links is True
    assert var_0.indented_import_headings is True
    assert var_0.honor_case_in_force_sorted_sections is False
    assert var_0.sort_relative_in_force_sorted_sections is False
    assert var_0.overwrite_in_place is False
    assert var_0.reverse_sort is False
    assert var_0.star_first is False
    assert var_0.git_ls_files == {}
    assert var_0.format_error == '{error}: {message}'
    assert var_0.format_success == '{success}: {message}'
    assert var_0.sort_order == 'natural'
    assert var_0.sort_reexports is False
    assert var_0.split_on_trailing_comma is False
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_0.Config.known_patterns).__module__}.{type(module_0.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments).__module__}.{type(module_0.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.section_comments_end).__module__}.{type(module_0.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skips).__module__}.{type(module_0.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.skip_globs).__module__}.{type(module_0.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_0.Config.sorting_function).__module__}.{type(module_0.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_1 = 'module.pyi'
    var_2 = var_0.is_supported_filetype(var_1)
    assert var_2 is True
    var_3 = 'YjWZ_sD 9 UBI;'
    var_4 = 'test.py~'
    var_5 = var_0.is_supported_filetype(var_4)
    assert var_5 is False
    var_6 = var_0.is_supported_filetype(var_3)
    assert var_6 is False
    var_7 = 'test.txt'
    var_8 = var_0.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = var_0.is_supported_filetype(var_7)
    assert var_9 is False
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = module_0.Config()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'isort.settings.Config'
    assert var_12.py_version == 'py3'
    assert f'{type(var_12.force_to_top).__module__}.{type(var_12.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.force_to_top) == 0
    assert f'{type(var_12.skip).__module__}.{type(var_12.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.skip) == 19
    assert f'{type(var_12.extend_skip).__module__}.{type(var_12.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.extend_skip) == 0
    assert f'{type(var_12.skip_glob).__module__}.{type(var_12.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.skip_glob) == 0
    assert f'{type(var_12.extend_skip_glob).__module__}.{type(var_12.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.extend_skip_glob) == 0
    assert var_12.skip_gitignore is False
    assert var_12.line_length == 79
    assert var_12.wrap_length == 0
    assert var_12.line_ending == ''
    assert var_12.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_12.no_sections is False
    assert f'{type(var_12.known_future_library).__module__}.{type(var_12.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_future_library) == 1
    assert f'{type(var_12.known_third_party).__module__}.{type(var_12.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_third_party) == 0
    assert f'{type(var_12.known_first_party).__module__}.{type(var_12.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_first_party) == 0
    assert f'{type(var_12.known_local_folder).__module__}.{type(var_12.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_local_folder) == 0
    assert f'{type(var_12.known_standard_library).__module__}.{type(var_12.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_standard_library) == 234
    assert f'{type(var_12.extra_standard_library).__module__}.{type(var_12.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.extra_standard_library) == 0
    assert var_12.known_other == {}
    assert var_12.multi_line_output == module_1.WrapModes.GRID
    assert var_12.forced_separate == ()
    assert var_12.indent == '    '
    assert var_12.comment_prefix == '  #'
    assert var_12.length_sort is False
    assert var_12.length_sort_straight is False
    assert f'{type(var_12.length_sort_sections).__module__}.{type(var_12.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.length_sort_sections) == 0
    assert f'{type(var_12.add_imports).__module__}.{type(var_12.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.add_imports) == 0
    assert f'{type(var_12.remove_imports).__module__}.{type(var_12.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.remove_imports) == 0
    assert var_12.append_only is False
    assert var_12.reverse_relative is False
    assert var_12.force_single_line is False
    assert var_12.single_line_exclusions == ()
    assert var_12.default_section == 'THIRDPARTY'
    assert var_12.import_headings == {}
    assert var_12.import_footers == {}
    assert var_12.balanced_wrapping is False
    assert var_12.use_parentheses is False
    assert var_12.order_by_type is True
    assert var_12.atomic is False
    assert var_12.lines_before_imports == -1
    assert var_12.lines_after_imports == -1
    assert var_12.lines_between_sections == 1
    assert var_12.lines_between_types == 0
    assert var_12.combine_as_imports is False
    assert var_12.combine_star is False
    assert var_12.include_trailing_comma is False
    assert var_12.from_first is False
    assert var_12.verbose is False
    assert var_12.quiet is False
    assert var_12.force_adds is False
    assert var_12.force_alphabetical_sort_within_sections is False
    assert var_12.force_alphabetical_sort is False
    assert var_12.force_grid_wrap == 0
    assert var_12.force_sort_within_sections is False
    assert var_12.lexicographical is False
    assert var_12.group_by_package is False
    assert var_12.ignore_whitespace is False
    assert f'{type(var_12.no_lines_before).__module__}.{type(var_12.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.no_lines_before) == 0
    assert var_12.no_inline_sort is False
    assert var_12.ignore_comments is False
    assert var_12.case_sensitive is False
    assert f'{type(var_12.sources).__module__}.{type(var_12.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_12.sources) == 1
    assert var_12.virtual_env == ''
    assert var_12.conda_env == ''
    assert var_12.ensure_newline_before_comments is False
    assert var_12.directory == '/workspace'
    assert var_12.profile == ''
    assert var_12.honor_noqa is False
    assert f'{type(var_12.src_paths).__module__}.{type(var_12.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_12.src_paths) == 2
    assert var_12.remove_redundant_aliases is False
    assert var_12.float_to_top is False
    assert var_12.filter_files is False
    assert var_12.formatter == ''
    assert var_12.formatting_function is None
    assert var_12.color_output is False
    assert f'{type(var_12.treat_comments_as_code).__module__}.{type(var_12.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.treat_comments_as_code) == 0
    assert var_12.treat_all_comments_as_code is False
    assert f'{type(var_12.supported_extensions).__module__}.{type(var_12.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.supported_extensions) == 4
    assert f'{type(var_12.blocked_extensions).__module__}.{type(var_12.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.blocked_extensions) == 1
    assert f'{type(var_12.constants).__module__}.{type(var_12.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.constants) == 0
    assert f'{type(var_12.classes).__module__}.{type(var_12.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.classes) == 0
    assert f'{type(var_12.variables).__module__}.{type(var_12.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.variables) == 0
    assert var_12.dedup_headings is False
    assert var_12.only_sections is False
    assert var_12.only_modified is False
    assert var_12.combine_straight_imports is False
    assert var_12.auto_identify_namespace_packages is True
    assert f'{type(var_12.namespace_packages).__module__}.{type(var_12.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.namespace_packages) == 0
    assert var_12.follow_links is True
    assert var_12.indented_import_headings is True
    assert var_12.honor_case_in_force_sorted_sections is False
    assert var_12.sort_relative_in_force_sorted_sections is False
    assert var_12.overwrite_in_place is False
    assert var_12.reverse_sort is False
    assert var_12.star_first is False
    assert var_12.git_ls_files == {}
    assert var_12.format_error == '{error}: {message}'
    assert var_12.format_success == '{success}: {message}'
    assert var_12.sort_order == 'natural'
    assert var_12.sort_reexports is False
    assert var_12.split_on_trailing_comma is False
    var_13 = var_12.is_supported_filetype(var_10)
    assert var_13 is False

def test_case_16():
    var_0 = '//'
    var_1 = module_0.find_all_configs(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_1.root).__module__}.{type(var_1.root).__qualname__}' == 'isort.utils.TrieNode'
    assert module_0.TYPE_CHECKING is False
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.FIRSTPARTY == 'FIRSTPARTY'
    assert module_0.FUTURE == 'FUTURE'
    assert module_0.LOCALFOLDER == 'LOCALFOLDER'
    assert module_0.STDLIB == 'STDLIB'
    assert module_0.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_0.SUPPORTED_EXTENSIONS).__module__}.{type(module_0.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_0.BLOCKED_EXTENSIONS).__module__}.{type(module_0.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.BLOCKED_EXTENSIONS) == 1
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_0.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_0.DEFAULT_SKIP).__module__}.{type(module_0.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_SKIP) == 19
    assert module_0.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_0.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_0.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_0.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_0.KNOWN_PREFIX == 'known_'
    assert module_0.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_0.RUNTIME_SOURCE == 'runtime'
    assert module_0.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False