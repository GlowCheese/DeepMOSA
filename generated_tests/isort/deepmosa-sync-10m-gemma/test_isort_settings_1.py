# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.settings as module_0
import isort.wrap_modes as module_1
import importlib.metadata as module_2
import isort.exceptions as module_3
import pathlib as module_4
import builtins as module_5
import posixpath as module_6
import enum as module_7

def test_case_0():
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
    assert var_0.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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

def test_case_1():
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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

def test_case_2():
    var_0 = '\x0cdH\x0bzCc"'
    with pytest.raises(module_3.InvalidSettingsPath):
        module_0.Config(settings_path=var_0)

def test_case_3():
    var_0 = '($L U\rgFHl\t'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '/fake/project/pyproject.toml'
    var_1 = {}
    module_0.Config(var_0, **var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '/fake/projece/pysrject.oml'
    var_1 = {var_0: var_0}
    module_0.Config(var_0, **var_1)

def test_case_6():
    var_0 = 'indent'
    var_1 = {var_0: var_0}
    var_2 = module_0.Config(**var_1)
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
    assert var_2.indent == 'indent'
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
    assert len(var_2.sources) == 2
    assert var_2.virtual_env == ''
    assert var_2.conda_env == ''
    assert var_2.ensure_newline_before_comments is False
    assert var_2.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_3 = var_2.indent

def test_case_7():
    var_0 = "inv'eTt"
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    with pytest.raises(module_3.UnsupportedSettings):
        module_0.Config(**var_1)

def test_case_8():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
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
    assert var_3.skip_gitignore is True
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
    assert var_3.multi_line_output == module_1.WrapModes.GRID
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
    assert var_3.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = var_3.is_supported_filetype(var_1)
    assert var_4 is False

def test_case_9():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
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
    assert var_3.skip_gitignore is True
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
    assert var_3.multi_line_output == module_1.WrapModes.GRID
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
    assert var_3.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = '/tmp'
    var_5 = [var_4]
    var_6 = {}
    var_7 = var_3.is_supported_filetype(var_4)
    assert var_7 is False
    var_8 = module_4.Path(*var_5, **var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_9 = module_0.find_all_configs(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_9.root).__module__}.{type(var_9.root).__qualname__}' == 'isort.utils.TrieNode'
    var_10 = []
    var_11 = module_4.Path(*var_10, **var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pathlib.PosixPath'
    var_12 = var_3.is_skipped(var_11)
    assert var_12 is False
    with pytest.raises(ValueError):
        var_3.__post_init__()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'qX("\nBkT-X:bV'
    var_1 = None
    var_2 = True
    var_3 = False
    var_4 = '6}#\\U8 i_dP7D'
    var_5 = module_5.dict
    var_6 = set()
    var_7 = 73
    var_8 = True
    var_9 = True
    var_10 = True
    var_11 = True
    var_12 = False
    var_13 = '=r'
    var_14 = module_0._Config(skip=var_5, skip_gitignore=var_1, sections=var_1, length_sort_sections=var_6, use_parentheses=var_2, atomic=var_1, lines_after_imports=var_7, lines_between_sections=var_7, combine_as_imports=var_8, verbose=var_2, quiet=var_9, force_alphabetical_sort_within_sections=var_1, force_alphabetical_sort=var_10, no_lines_before=var_3, no_inline_sort=var_2, case_sensitive=var_10, virtual_env=var_4, conda_env=var_0, honor_noqa=var_10, color_output=var_3, treat_all_comments_as_code=var_11, classes=var_1, combine_straight_imports=var_11, indented_import_headings=var_12, honor_case_in_force_sorted_sections=var_1, format_success=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'isort.settings._Config'
    assert var_14.py_version == 'py3'
    assert f'{type(var_14.force_to_top).__module__}.{type(var_14.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.force_to_top) == 0
    assert f'{type(var_14.extend_skip).__module__}.{type(var_14.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extend_skip) == 0
    assert f'{type(var_14.skip_glob).__module__}.{type(var_14.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.skip_glob) == 0
    assert f'{type(var_14.extend_skip_glob).__module__}.{type(var_14.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extend_skip_glob) == 0
    assert var_14.skip_gitignore is None
    assert var_14.line_length == 79
    assert var_14.wrap_length == 0
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
    assert f'{type(var_14.extra_standard_library).__module__}.{type(var_14.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extra_standard_library) == 0
    assert var_14.known_other == {}
    assert var_14.multi_line_output == module_1.WrapModes.GRID
    assert var_14.forced_separate == ()
    assert var_14.indent == '    '
    assert var_14.comment_prefix == '  #'
    assert var_14.length_sort is False
    assert var_14.length_sort_straight is False
    assert var_14.length_sort_sections == {*()}
    assert f'{type(var_14.add_imports).__module__}.{type(var_14.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.add_imports) == 0
    assert f'{type(var_14.remove_imports).__module__}.{type(var_14.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.remove_imports) == 0
    assert var_14.append_only is False
    assert var_14.reverse_relative is False
    assert var_14.force_single_line is False
    assert var_14.single_line_exclusions == ()
    assert var_14.default_section == 'THIRDPARTY'
    assert var_14.import_headings == {}
    assert var_14.import_footers == {}
    assert var_14.balanced_wrapping is False
    assert var_14.use_parentheses is True
    assert var_14.order_by_type is True
    assert var_14.atomic is None
    assert var_14.lines_before_imports == -1
    assert var_14.lines_after_imports == 73
    assert var_14.lines_between_sections == 73
    assert var_14.lines_between_types == 1
    assert var_14.combine_as_imports is True
    assert var_14.combine_star is False
    assert var_14.include_trailing_comma is False
    assert var_14.from_first is True
    assert var_14.verbose is True
    assert var_14.quiet is True
    assert var_14.force_adds is False
    assert var_14.force_alphabetical_sort_within_sections is True
    assert var_14.force_alphabetical_sort is True
    assert var_14.force_grid_wrap == 0
    assert var_14.force_sort_within_sections is False
    assert var_14.lexicographical is False
    assert var_14.group_by_package is False
    assert var_14.ignore_whitespace is False
    assert var_14.no_lines_before is False
    assert var_14.no_inline_sort is True
    assert var_14.ignore_comments is False
    assert var_14.case_sensitive is True
    assert var_14.sources == ()
    assert var_14.virtual_env == '6}#\\U8 i_dP7D'
    assert var_14.conda_env == 'qX("\nBkT-X:bV'
    assert var_14.ensure_newline_before_comments is False
    assert var_14.directory == ''
    assert var_14.profile == ''
    assert var_14.honor_noqa is True
    assert var_14.src_paths == ()
    assert var_14.remove_redundant_aliases is False
    assert var_14.float_to_top is False
    assert var_14.filter_files is False
    assert var_14.formatter == ''
    assert var_14.formatting_function is None
    assert var_14.color_output is False
    assert f'{type(var_14.treat_comments_as_code).__module__}.{type(var_14.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.treat_comments_as_code) == 0
    assert var_14.treat_all_comments_as_code is True
    assert f'{type(var_14.supported_extensions).__module__}.{type(var_14.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.supported_extensions) == 4
    assert f'{type(var_14.blocked_extensions).__module__}.{type(var_14.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.blocked_extensions) == 1
    assert f'{type(var_14.constants).__module__}.{type(var_14.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.constants) == 0
    assert var_14.classes is None
    assert f'{type(var_14.variables).__module__}.{type(var_14.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.variables) == 0
    assert var_14.dedup_headings is False
    assert var_14.only_sections is False
    assert var_14.only_modified is False
    assert var_14.combine_straight_imports is True
    assert var_14.auto_identify_namespace_packages is True
    assert f'{type(var_14.namespace_packages).__module__}.{type(var_14.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.namespace_packages) == 0
    assert var_14.follow_links is True
    assert var_14.indented_import_headings is False
    assert var_14.honor_case_in_force_sorted_sections is None
    assert var_14.sort_relative_in_force_sorted_sections is False
    assert var_14.overwrite_in_place is False
    assert var_14.reverse_sort is False
    assert var_14.star_first is False
    assert var_14.git_ls_files == {}
    assert var_14.format_error == '{error}: {message}'
    assert var_14.format_success == '=r'
    assert var_14.sort_order == 'natural'
    assert var_14.sort_reexports is False
    assert var_14.split_on_trailing_comma is False
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    module_6.dirname(var_1)

def test_case_11():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
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
    assert var_3.skip_gitignore is True
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
    assert var_3.multi_line_output == module_1.WrapModes.GRID
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
    assert var_3.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = var_3.is_supported_filetype(var_1)
    assert var_4 is False
    var_5 = [var_1]
    var_6 = {}
    var_7 = 'pbyDUu.{'
    var_8 = var_3.is_supported_filetype(var_7)
    assert var_8 is False
    var_9 = module_4.Path(*var_5, **var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_10 = '/tmp/tracked_file.py'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_4.Path(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pathlib.PosixPath'
    var_14 = var_3.is_skipped(var_13)
    assert var_14 is True
    var_15 = var_3.is_skipped(var_9)
    assert var_15 is True
    var_16 = module_0.Config(config=var_3)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'isort.settings.Config'
    assert var_16.py_version == 'py3'
    assert f'{type(var_16.force_to_top).__module__}.{type(var_16.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.force_to_top) == 0
    assert f'{type(var_16.skip).__module__}.{type(var_16.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.skip) == 19
    assert f'{type(var_16.extend_skip).__module__}.{type(var_16.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.extend_skip) == 0
    assert f'{type(var_16.skip_glob).__module__}.{type(var_16.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.skip_glob) == 0
    assert f'{type(var_16.extend_skip_glob).__module__}.{type(var_16.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.extend_skip_glob) == 0
    assert var_16.skip_gitignore is True
    assert var_16.line_length == 79
    assert var_16.wrap_length == 0
    assert var_16.line_ending == ''
    assert var_16.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_16.no_sections is False
    assert f'{type(var_16.known_future_library).__module__}.{type(var_16.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.known_future_library) == 1
    assert f'{type(var_16.known_third_party).__module__}.{type(var_16.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.known_third_party) == 0
    assert f'{type(var_16.known_first_party).__module__}.{type(var_16.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.known_first_party) == 0
    assert f'{type(var_16.known_local_folder).__module__}.{type(var_16.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.known_local_folder) == 0
    assert f'{type(var_16.known_standard_library).__module__}.{type(var_16.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.known_standard_library) == 234
    assert f'{type(var_16.extra_standard_library).__module__}.{type(var_16.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.extra_standard_library) == 0
    assert var_16.known_other == {}
    assert var_16.multi_line_output == module_1.WrapModes.GRID
    assert var_16.forced_separate == ()
    assert var_16.indent == '    '
    assert var_16.comment_prefix == '  #'
    assert var_16.length_sort is False
    assert var_16.length_sort_straight is False
    assert f'{type(var_16.length_sort_sections).__module__}.{type(var_16.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.length_sort_sections) == 0
    assert f'{type(var_16.add_imports).__module__}.{type(var_16.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.add_imports) == 0
    assert f'{type(var_16.remove_imports).__module__}.{type(var_16.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.remove_imports) == 0
    assert var_16.append_only is False
    assert var_16.reverse_relative is False
    assert var_16.force_single_line is False
    assert var_16.single_line_exclusions == ()
    assert var_16.default_section == 'THIRDPARTY'
    assert var_16.import_headings == {}
    assert var_16.import_footers == {}
    assert var_16.balanced_wrapping is False
    assert var_16.use_parentheses is False
    assert var_16.order_by_type is True
    assert var_16.atomic is False
    assert var_16.lines_before_imports == -1
    assert var_16.lines_after_imports == -1
    assert var_16.lines_between_sections == 1
    assert var_16.lines_between_types == 0
    assert var_16.combine_as_imports is False
    assert var_16.combine_star is False
    assert var_16.include_trailing_comma is False
    assert var_16.from_first is False
    assert var_16.verbose is False
    assert var_16.quiet is False
    assert var_16.force_adds is False
    assert var_16.force_alphabetical_sort_within_sections is False
    assert var_16.force_alphabetical_sort is False
    assert var_16.force_grid_wrap == 0
    assert var_16.force_sort_within_sections is False
    assert var_16.lexicographical is False
    assert var_16.group_by_package is False
    assert var_16.ignore_whitespace is False
    assert f'{type(var_16.no_lines_before).__module__}.{type(var_16.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.no_lines_before) == 0
    assert var_16.no_inline_sort is False
    assert var_16.ignore_comments is False
    assert var_16.case_sensitive is False
    assert f'{type(var_16.sources).__module__}.{type(var_16.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_16.sources) == 2
    assert var_16.virtual_env == ''
    assert var_16.conda_env == ''
    assert var_16.ensure_newline_before_comments is False
    assert var_16.directory == '/workspace/run'
    assert var_16.profile == ''
    assert var_16.honor_noqa is False
    assert f'{type(var_16.src_paths).__module__}.{type(var_16.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_16.src_paths) == 2
    assert var_16.remove_redundant_aliases is False
    assert var_16.float_to_top is False
    assert var_16.filter_files is False
    assert var_16.formatter == ''
    assert var_16.formatting_function is None
    assert var_16.color_output is False
    assert f'{type(var_16.treat_comments_as_code).__module__}.{type(var_16.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.treat_comments_as_code) == 0
    assert var_16.treat_all_comments_as_code is False
    assert f'{type(var_16.supported_extensions).__module__}.{type(var_16.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.supported_extensions) == 4
    assert f'{type(var_16.blocked_extensions).__module__}.{type(var_16.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.blocked_extensions) == 1
    assert f'{type(var_16.constants).__module__}.{type(var_16.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.constants) == 0
    assert f'{type(var_16.classes).__module__}.{type(var_16.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.classes) == 0
    assert f'{type(var_16.variables).__module__}.{type(var_16.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.variables) == 0
    assert var_16.dedup_headings is False
    assert var_16.only_sections is False
    assert var_16.only_modified is False
    assert var_16.combine_straight_imports is False
    assert var_16.auto_identify_namespace_packages is True
    assert f'{type(var_16.namespace_packages).__module__}.{type(var_16.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_16.namespace_packages) == 0
    assert var_16.follow_links is True
    assert var_16.indented_import_headings is True
    assert var_16.honor_case_in_force_sorted_sections is False
    assert var_16.sort_relative_in_force_sorted_sections is False
    assert var_16.overwrite_in_place is False
    assert var_16.reverse_sort is False
    assert var_16.star_first is False
    assert var_16.git_ls_files == {}
    assert var_16.format_error == '{error}: {message}'
    assert var_16.format_success == '{success}: {message}'
    assert var_16.sort_order == 'natural'
    assert var_16.sort_reexports is False
    assert var_16.split_on_trailing_comma is False

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'skip_glob'
    var_1 = {var_0: var_0}
    var_2 = module_0.Config(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.settings.Config'
    assert var_2.py_version == 'py3'
    assert f'{type(var_2.force_to_top).__module__}.{type(var_2.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.force_to_top) == 0
    assert f'{type(var_2.skip).__module__}.{type(var_2.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.skip) == 19
    assert f'{type(var_2.extend_skip).__module__}.{type(var_2.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.extend_skip) == 0
    assert f'{type(var_2.skip_glob).__module__}.{type(var_2.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.skip_glob) == 9
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
    assert len(var_2.sources) == 2
    assert var_2.virtual_env == ''
    assert var_2.conda_env == ''
    assert var_2.ensure_newline_before_comments is False
    assert var_2.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_3 = var_2.__hash__()
    assert var_3 == 129411079516992
    var_2.is_skipped(var_2)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
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
    assert var_1.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = None
    var_1.is_skipped(var_2)

@pytest.mark.xfail(strict=True)
def test_case_14():
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = False
    var_3 = False
    var_4 = True
    var_5 = False
    var_6 = True
    var_7 = False
    var_8 = True
    module_0._Config(extend_skip_glob=var_0, skip_gitignore=var_0, line_length=var_3, wrap_length=var_4, known_third_party=var_0, forced_separate=var_0, length_sort=var_0, reverse_relative=var_0, import_headings=var_0, import_footers=var_0, lines_before_imports=var_0, combine_as_imports=var_5, quiet=var_2, force_alphabetical_sort=var_0, group_by_package=var_5, no_lines_before=var_0, ignore_comments=var_0, virtual_env=var_0, remove_redundant_aliases=var_6, formatter=var_0, formatting_function=var_0, treat_comments_as_code=var_0, blocked_extensions=var_0, only_sections=var_0, only_modified=var_0, auto_identify_namespace_packages=var_7, follow_links=var_0, honor_case_in_force_sorted_sections=var_8, split_on_trailing_comma=var_6)

def test_case_15():
    var_0 = 'skip_gitignore'
    var_1 = {var_0: var_0}
    var_2 = module_0.Config(**var_1)
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
    assert var_2.skip_gitignore is True
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
    assert len(var_2.sources) == 2
    assert var_2.virtual_env == ''
    assert var_2.conda_env == ''
    assert var_2.ensure_newline_before_comments is False
    assert var_2.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_3 = '/tmp'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_4.Path(*var_4, **var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_7 = var_2.is_skipped(var_6)
    assert var_7 is False

def test_case_16():
    var_0 = module_7._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.Config(**var_0)
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
    assert var_1.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = []
    var_3 = module_4.Path(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_4 = var_1.is_skipped(var_3)
    assert var_4 is False

def test_case_17():
    var_0 = module_7._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.Config(**var_0)
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
    assert var_1.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = []
    var_3 = module_4.Path(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_4 = var_1.is_skipped(var_3)
    assert var_4 is False
    var_5 = var_1.is_skipped(var_3)
    assert var_5 is False

def test_case_18():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
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
    assert var_1.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = 'Z?| 2*BZ>7m;%;\t\x0b'
    var_3 = [var_2]
    var_4 = module_4.Path(*var_3, **var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_5 = var_1.is_skipped(var_4)
    assert var_5 is True

def test_case_19():
    var_0 = 'skip_glob'
    var_1 = {var_0: var_0}
    var_2 = module_0.Config(**var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.settings.Config'
    assert var_2.py_version == 'py3'
    assert f'{type(var_2.force_to_top).__module__}.{type(var_2.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.force_to_top) == 0
    assert f'{type(var_2.skip).__module__}.{type(var_2.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.skip) == 19
    assert f'{type(var_2.extend_skip).__module__}.{type(var_2.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.extend_skip) == 0
    assert f'{type(var_2.skip_glob).__module__}.{type(var_2.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.skip_glob) == 9
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
    assert len(var_2.sources) == 2
    assert var_2.virtual_env == ''
    assert var_2.conda_env == ''
    assert var_2.ensure_newline_before_comments is False
    assert var_2.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_3 = module_6.expanduser(var_0)
    assert var_3 == 'skip_glob'
    assert module_6.curdir == '.'
    assert module_6.pardir == '..'
    assert module_6.extsep == '.'
    assert module_6.sep == '/'
    assert module_6.pathsep == ':'
    assert module_6.defpath == '/bin:/usr/bin'
    assert module_6.altsep is None
    assert module_6.devnull == '/dev/null'
    assert f'{type(module_6.ALLOW_MISSING).__module__}.{type(module_6.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    assert module_6.supports_unicode_filenames is False
    var_4 = {}
    var_5 = module_4.Path(*var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_6 = var_2.is_skipped(var_5)
    assert var_6 is True

def test_case_20():
    var_0 = '*.tmp'
    var_1 = [var_0]
    var_2 = 'skip_glob'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.settings.Config'
    assert var_4.py_version == 'py3'
    assert f'{type(var_4.force_to_top).__module__}.{type(var_4.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.force_to_top) == 0
    assert f'{type(var_4.skip).__module__}.{type(var_4.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.skip) == 19
    assert f'{type(var_4.extend_skip).__module__}.{type(var_4.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.extend_skip) == 0
    assert f'{type(var_4.skip_glob).__module__}.{type(var_4.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.skip_glob) == 1
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
    assert len(var_4.sources) == 2
    assert var_4.virtual_env == ''
    assert var_4.conda_env == ''
    assert var_4.ensure_newline_before_comments is False
    assert var_4.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_5 = 'data.tmp'
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_4.Path(*var_6, **var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_9 = var_4.is_skipped(var_8)
    assert var_9 is True
    var_10 = 'subdir/temp_file.tmp'
    var_11 = [var_10]
    var_12 = {}
    var_13 = module_4.Path(*var_11, **var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pathlib.PosixPath'
    var_14 = var_4.is_skipped(var_8)
    assert var_14 is True

def test_case_21():
    var_0 = 'profile'
    var_1 = {var_0: var_0}
    with pytest.raises(module_3.ProfileDoesNotExist):
        module_0.Config(**var_1)

def test_case_22():
    var_0 = 'test_file.py'
    var_1 = [var_0]
    var_2 = 'skip'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.settings.Config'
    assert var_4.py_version == 'py3'
    assert f'{type(var_4.force_to_top).__module__}.{type(var_4.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.force_to_top) == 0
    assert f'{type(var_4.skip).__module__}.{type(var_4.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.skip) == 1
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
    assert len(var_4.sources) == 2
    assert var_4.virtual_env == ''
    assert var_4.conda_env == ''
    assert var_4.ensure_newline_before_comments is False
    assert var_4.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_5 = [var_0]
    var_6 = module_4.Path(*var_5, **var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_7 = var_4.is_skipped(var_6)
    assert var_7 is True

def test_case_23():
    var_0 = '4'
    var_1 = 'indent'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
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
    assert var_3.multi_line_output == module_1.WrapModes.GRID
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
    assert var_3.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = var_3.indent
    assert var_4 == '    '

def test_case_24():
    var_0 = 'tab'
    var_1 = 'indent'
    var_2 = {var_1: var_1, var_1: var_0, var_0: var_0}
    with pytest.raises(module_3.UnsupportedSettings):
        module_0.Config(**var_2)

def test_case_25():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0}
    var_3 = module_0.Config(**var_2)
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
    assert var_3.skip_gitignore is True
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
    assert var_3.multi_line_output == module_1.WrapModes.GRID
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
    assert var_3.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = '/tmp'
    var_5 = var_3.is_supported_filetype(var_4)
    assert var_5 is False
    var_6 = module_0.find_all_configs(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_6.root).__module__}.{type(var_6.root).__qualname__}' == 'isort.utils.TrieNode'
    var_7 = '/tmp/tracked_file.py'
    var_8 = [var_7]
    var_9 = {}
    var_10 = module_4.Path(*var_8, **var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = True
    var_1 = '/tmp'
    var_2 = [var_1]
    var_3 = module_0.find_all_configs(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.utils.Trie'
    assert f'{type(var_3.root).__module__}.{type(var_3.root).__qualname__}' == 'isort.utils.TrieNode'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = [var_0]
    var_5 = var_2.__iter__()
    var_6 = {}
    module_4.Path(*var_4, **var_6)

def test_case_27():
    var_0 = '/home/user/project'
    var_1 = module_0._find_config(var_0)
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = bool(var_1 == ('/home/user/project', {}))
    assert var_2 is True

def test_case_28():
    var_0 = '/tmp'
    var_1 = None
    var_2 = 'C`Vbp'
    var_3 = {var_2: var_1, var_0: var_1, var_2: var_1}
    with pytest.raises(module_3.UnsupportedSettings):
        module_0.Config(settings_path=var_0, **var_3)

def test_case_29():
    var_0 = 'sme_module'
    var_1 = [var_0]
    var_2 = 'known_custom'
    var_3 = {var_2: var_1}
    var_4 = module_0.Config(**var_3)
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
    assert f'{type(var_4.known_other).__module__}.{type(var_4.known_other).__qualname__}' == 'builtins.dict'
    assert len(var_4.known_other) == 1
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
    assert len(var_4.sources) == 2
    assert var_4.virtual_env == ''
    assert var_4.conda_env == ''
    assert var_4.ensure_newline_before_comments is False
    assert var_4.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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

def test_case_30():
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = module_0._find_config(var_1)

def test_case_31():
    var_0 = {}
    var_1 = module_0.Config(**var_0)
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
    assert var_1.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = 'test.py'
    var_3 = var_1.is_supported_filetype(var_2)
    assert var_3 is True

def test_case_32():
    var_0 = True
    var_1 = 'skip_gitignore'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_0.Config(**var_2)
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
    assert var_3.skip_gitignore is True
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
    assert var_3.multi_line_output == module_1.WrapModes.GRID
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
    assert var_3.directory == '/workspace/run'
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = '.'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_4.Path(*var_5, **var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'pathlib.PosixPath'
    assert module_4.EINVAL == 22
    assert module_4.ENOENT == 2
    assert module_4.ENOTDIR == 20
    assert module_4.EBADF == 9
    assert module_4.ELOOP == 40
    var_8 = '.git/config'
    var_9 = [var_8]
    var_10 = {}
    var_11 = module_4.Path(*var_9, **var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'pathlib.PosixPath'
    var_12 = var_3.is_skipped(var_11)
    assert var_12 is True

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = 'X\x0cb^\x0cZ('
    var_1 = module_0._find_config(var_0)
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = None
    module_6.splitext(var_2)