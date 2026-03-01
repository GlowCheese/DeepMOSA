# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.core as module_0
import urllib.request as module_1
import email._header_value_parser as module_2
import _io as module_3
import isort.settings as module_4
import isort.wrap_modes as module_5

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'snG#+[)TI'
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1.noheaders()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'email.message.Message'
    assert len(var_0) == 2
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_2.quote_string(var_0)
    assert var_1 == '"None"'
    assert module_2.hexdigits == '0123456789abcdefABCDEF'
    assert module_2.WSP == {'\t', ' '}
    assert module_2.CFWS_LEADER == {'\t', '(', ' '}
    assert module_2.SPECIALS == {'@', '\\', '"', '.', ']', ',', '(', '>', ')', '[', '<', ':', ';'}
    assert module_2.ATOM_ENDS == {'@', '\\', ' ', '.', '"', ']', ',', '(', '>', ')', '[', '\t', '<', ':', ';'}
    assert module_2.DOT_ATOM_ENDS == {'@', '\\', ' ', '"', ']', ',', '(', '>', ')', '[', '\t', '<', ':', ';'}
    assert module_2.PHRASE_ENDS == {'@', '\\', ';', ']', ',', '[', '>', ')', ':', '<'}
    assert module_2.TSPECIALS == {'\\', '@', '=', ';', '?', '"', ']', ',', '[', '(', '>', ')', '/', ':', '<'}
    assert module_2.TOKEN_ENDS == {'@', '\\', '=', '?', ' ', '"', ']', ',', '[', '(', '>', ')', '\t', '/', '<', ':', ';'}
    assert module_2.ASPECIALS == {'@', "'", '\\', '=', ':', '?', '*', '"', ']', ',', '[', '(', '>', ')', '/', '<', '%', ';'}
    assert module_2.ATTRIBUTE_ENDS == {"'", ']', '[', '\t', '%', '@', '=', '?', '*', '"', '>', '/', ':', ' ', '<', '\\', ',', '(', ')', ';'}
    assert module_2.EXTENDED_ATTRIBUTE_ENDS == {"'", ']', '[', '\t', '@', '=', '?', '*', '"', '>', '/', ':', ' ', '<', '\\', ',', '(', ')', ';'}
    assert module_2.NLSET == {'\n', '\r'}
    assert module_2.SPECIALSNL == {'@', '\\', '\n', '.', '"', ']', ',', '(', '>', ')', '[', '<', ':', '\r', ';'}
    assert f'{type(module_2.rfc2047_matcher).__module__}.{type(module_2.rfc2047_matcher).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.DOT).__module__}.{type(module_2.DOT).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_2.DOT) == 1
    assert f'{type(module_2.ListSeparator).__module__}.{type(module_2.ListSeparator).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_2.ListSeparator) == 1
    assert f'{type(module_2.RouteComponentMarker).__module__}.{type(module_2.RouteComponentMarker).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_2.RouteComponentMarker) == 1
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = ' _9xA44O1],r=aOI`Y'
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = 'nG#+[)T:I'
    var_2 = var_1.__repr__()
    assert var_2 == "'nG#+[)T:I'"
    module_0.process(var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '#snG#+/)TI'
    module_0.process(var_0, var_0)

def test_case_7():
    var_0 = module_3.StringIO()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == '_io.StringIO'
    assert module_3.DEFAULT_BUFFER_SIZE == 8192
    assert f'{type(module_3.StringIO.closed).__module__}.{type(module_3.StringIO.closed).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.StringIO.newlines).__module__}.{type(module_3.StringIO.newlines).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_3.StringIO.line_buffering).__module__}.{type(module_3.StringIO.line_buffering).__qualname__}' == 'builtins.getset_descriptor'
    var_1 = module_3.StringIO()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == '_io.StringIO'
    var_2 = module_3.StringIO()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == '_io.StringIO'
    var_3 = 'import added'
    var_4 = [var_3]
    var_5 = module_4.Config()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.settings.Config'
    assert var_5.py_version == 'py3'
    assert f'{type(var_5.force_to_top).__module__}.{type(var_5.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.force_to_top) == 0
    assert f'{type(var_5.skip).__module__}.{type(var_5.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.skip) == 19
    assert f'{type(var_5.extend_skip).__module__}.{type(var_5.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.extend_skip) == 0
    assert f'{type(var_5.skip_glob).__module__}.{type(var_5.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.skip_glob) == 0
    assert f'{type(var_5.extend_skip_glob).__module__}.{type(var_5.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.extend_skip_glob) == 0
    assert var_5.skip_gitignore is False
    assert var_5.line_length == 79
    assert var_5.wrap_length == 0
    assert var_5.line_ending == ''
    assert var_5.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_5.no_sections is False
    assert f'{type(var_5.known_future_library).__module__}.{type(var_5.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_future_library) == 1
    assert f'{type(var_5.known_third_party).__module__}.{type(var_5.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_third_party) == 0
    assert f'{type(var_5.known_first_party).__module__}.{type(var_5.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_first_party) == 0
    assert f'{type(var_5.known_local_folder).__module__}.{type(var_5.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_local_folder) == 0
    assert f'{type(var_5.known_standard_library).__module__}.{type(var_5.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_standard_library) == 234
    assert f'{type(var_5.extra_standard_library).__module__}.{type(var_5.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.extra_standard_library) == 0
    assert var_5.known_other == {}
    assert var_5.multi_line_output == module_5.WrapModes.GRID
    assert var_5.forced_separate == ()
    assert var_5.indent == '    '
    assert var_5.comment_prefix == '  #'
    assert var_5.length_sort is False
    assert var_5.length_sort_straight is False
    assert f'{type(var_5.length_sort_sections).__module__}.{type(var_5.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.length_sort_sections) == 0
    assert f'{type(var_5.add_imports).__module__}.{type(var_5.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.add_imports) == 0
    assert f'{type(var_5.remove_imports).__module__}.{type(var_5.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.remove_imports) == 0
    assert var_5.append_only is False
    assert var_5.reverse_relative is False
    assert var_5.force_single_line is False
    assert var_5.single_line_exclusions == ()
    assert var_5.default_section == 'THIRDPARTY'
    assert var_5.import_headings == {}
    assert var_5.import_footers == {}
    assert var_5.balanced_wrapping is False
    assert var_5.use_parentheses is False
    assert var_5.order_by_type is True
    assert var_5.atomic is False
    assert var_5.lines_before_imports == -1
    assert var_5.lines_after_imports == -1
    assert var_5.lines_between_sections == 1
    assert var_5.lines_between_types == 0
    assert var_5.combine_as_imports is False
    assert var_5.combine_star is False
    assert var_5.include_trailing_comma is False
    assert var_5.from_first is False
    assert var_5.verbose is False
    assert var_5.quiet is False
    assert var_5.force_adds is False
    assert var_5.force_alphabetical_sort_within_sections is False
    assert var_5.force_alphabetical_sort is False
    assert var_5.force_grid_wrap == 0
    assert var_5.force_sort_within_sections is False
    assert var_5.lexicographical is False
    assert var_5.group_by_package is False
    assert var_5.ignore_whitespace is False
    assert f'{type(var_5.no_lines_before).__module__}.{type(var_5.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.no_lines_before) == 0
    assert var_5.no_inline_sort is False
    assert var_5.ignore_comments is False
    assert var_5.case_sensitive is False
    assert f'{type(var_5.sources).__module__}.{type(var_5.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_5.sources) == 1
    assert var_5.virtual_env == ''
    assert var_5.conda_env == ''
    assert var_5.ensure_newline_before_comments is False
    assert var_5.directory == '/workspace'
    assert var_5.profile == ''
    assert var_5.honor_noqa is False
    assert f'{type(var_5.src_paths).__module__}.{type(var_5.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_5.src_paths) == 2
    assert var_5.remove_redundant_aliases is False
    assert var_5.float_to_top is False
    assert var_5.filter_files is False
    assert var_5.formatter == ''
    assert var_5.formatting_function is None
    assert var_5.color_output is False
    assert f'{type(var_5.treat_comments_as_code).__module__}.{type(var_5.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.treat_comments_as_code) == 0
    assert var_5.treat_all_comments_as_code is False
    assert f'{type(var_5.supported_extensions).__module__}.{type(var_5.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.supported_extensions) == 4
    assert f'{type(var_5.blocked_extensions).__module__}.{type(var_5.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.blocked_extensions) == 1
    assert f'{type(var_5.constants).__module__}.{type(var_5.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.constants) == 0
    assert f'{type(var_5.classes).__module__}.{type(var_5.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.classes) == 0
    assert f'{type(var_5.variables).__module__}.{type(var_5.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.variables) == 0
    assert var_5.dedup_headings is False
    assert var_5.only_sections is False
    assert var_5.only_modified is False
    assert var_5.combine_straight_imports is False
    assert var_5.auto_identify_namespace_packages is True
    assert f'{type(var_5.namespace_packages).__module__}.{type(var_5.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.namespace_packages) == 0
    assert var_5.follow_links is True
    assert var_5.indented_import_headings is True
    assert var_5.honor_case_in_force_sorted_sections is False
    assert var_5.sort_relative_in_force_sorted_sections is False
    assert var_5.overwrite_in_place is False
    assert var_5.reverse_sort is False
    assert var_5.star_first is False
    assert var_5.git_ls_files == {}
    assert var_5.format_error == '{error}: {message}'
    assert var_5.format_success == '{success}: {message}'
    assert var_5.sort_order == 'natural'
    assert var_5.sort_reexports is False
    assert var_5.split_on_trailing_comma is False
    assert module_4.TYPE_CHECKING is False
    assert module_4.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_4.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_4.FIRSTPARTY == 'FIRSTPARTY'
    assert module_4.FUTURE == 'FUTURE'
    assert module_4.LOCALFOLDER == 'LOCALFOLDER'
    assert module_4.STDLIB == 'STDLIB'
    assert module_4.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_4.CYTHON_EXTENSIONS).__module__}.{type(module_4.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_4.SUPPORTED_EXTENSIONS).__module__}.{type(module_4.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_4.BLOCKED_EXTENSIONS).__module__}.{type(module_4.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.BLOCKED_EXTENSIONS) == 1
    assert module_4.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_4.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_4.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_4.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_4.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_4.DEFAULT_SKIP).__module__}.{type(module_4.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_SKIP) == 19
    assert module_4.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_4.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_4.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_4.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_4.KNOWN_PREFIX == 'known_'
    assert module_4.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_4.RUNTIME_SOURCE == 'runtime'
    assert module_4.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_4.DEFAULT_CONFIG).__module__}.{type(module_4.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_4.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_4.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_4.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.skip).__module__}.{type(module_4.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_4.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_4.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_4.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_4.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_4.DEFAULT_CONFIG.skip_gitignore is False
    assert module_4.DEFAULT_CONFIG.line_length == 79
    assert module_4.DEFAULT_CONFIG.wrap_length == 0
    assert module_4.DEFAULT_CONFIG.line_ending == ''
    assert module_4.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_4.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_4.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_4.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_4.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_4.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_4.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_4.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_4.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_4.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_4.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_4.DEFAULT_CONFIG.known_other == {}
    assert module_4.DEFAULT_CONFIG.multi_line_output == module_5.WrapModes.GRID
    assert module_4.DEFAULT_CONFIG.forced_separate == ()
    assert module_4.DEFAULT_CONFIG.indent == '    '
    assert module_4.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_4.DEFAULT_CONFIG.length_sort is False
    assert module_4.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_4.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_4.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.add_imports).__module__}.{type(module_4.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_4.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.remove_imports) == 0
    assert module_4.DEFAULT_CONFIG.append_only is False
    assert module_4.DEFAULT_CONFIG.reverse_relative is False
    assert module_4.DEFAULT_CONFIG.force_single_line is False
    assert module_4.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_4.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_4.DEFAULT_CONFIG.import_headings == {}
    assert module_4.DEFAULT_CONFIG.import_footers == {}
    assert module_4.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_4.DEFAULT_CONFIG.use_parentheses is False
    assert module_4.DEFAULT_CONFIG.order_by_type is True
    assert module_4.DEFAULT_CONFIG.atomic is False
    assert module_4.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_4.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_4.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_4.DEFAULT_CONFIG.lines_between_types == 0
    assert module_4.DEFAULT_CONFIG.combine_as_imports is False
    assert module_4.DEFAULT_CONFIG.combine_star is False
    assert module_4.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_4.DEFAULT_CONFIG.from_first is False
    assert module_4.DEFAULT_CONFIG.verbose is False
    assert module_4.DEFAULT_CONFIG.quiet is False
    assert module_4.DEFAULT_CONFIG.force_adds is False
    assert module_4.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_4.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_4.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_4.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_4.DEFAULT_CONFIG.lexicographical is False
    assert module_4.DEFAULT_CONFIG.group_by_package is False
    assert module_4.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_4.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_4.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_4.DEFAULT_CONFIG.no_inline_sort is False
    assert module_4.DEFAULT_CONFIG.ignore_comments is False
    assert module_4.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_4.DEFAULT_CONFIG.sources).__module__}.{type(module_4.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_4.DEFAULT_CONFIG.sources) == 1
    assert module_4.DEFAULT_CONFIG.virtual_env == ''
    assert module_4.DEFAULT_CONFIG.conda_env == ''
    assert module_4.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_4.DEFAULT_CONFIG.directory == '/workspace'
    assert module_4.DEFAULT_CONFIG.profile == ''
    assert module_4.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_4.DEFAULT_CONFIG.src_paths).__module__}.{type(module_4.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_4.DEFAULT_CONFIG.src_paths) == 2
    assert module_4.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_4.DEFAULT_CONFIG.float_to_top is False
    assert module_4.DEFAULT_CONFIG.filter_files is False
    assert module_4.DEFAULT_CONFIG.formatter == ''
    assert module_4.DEFAULT_CONFIG.formatting_function is None
    assert module_4.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_4.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_4.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_4.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_4.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_4.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_4.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_4.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_4.DEFAULT_CONFIG.constants).__module__}.{type(module_4.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.classes).__module__}.{type(module_4.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.variables).__module__}.{type(module_4.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.variables) == 0
    assert module_4.DEFAULT_CONFIG.dedup_headings is False
    assert module_4.DEFAULT_CONFIG.only_sections is False
    assert module_4.DEFAULT_CONFIG.only_modified is False
    assert module_4.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_4.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_4.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_4.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_4.DEFAULT_CONFIG.follow_links is True
    assert module_4.DEFAULT_CONFIG.indented_import_headings is True
    assert module_4.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_4.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_4.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_4.DEFAULT_CONFIG.reverse_sort is False
    assert module_4.DEFAULT_CONFIG.star_first is False
    assert module_4.DEFAULT_CONFIG.git_ls_files == {}
    assert module_4.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_4.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_4.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_4.DEFAULT_CONFIG.sort_reexports is False
    assert module_4.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_4.Config.known_patterns).__module__}.{type(module_4.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.section_comments).__module__}.{type(module_4.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.section_comments_end).__module__}.{type(module_4.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.skips).__module__}.{type(module_4.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.skip_globs).__module__}.{type(module_4.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.sorting_function).__module__}.{type(module_4.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_6 = module_3.StringIO()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == '_io.StringIO'
    var_7 = module_3.StringIO()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == '_io.StringIO'
    var_8 = True
    var_9 = module_4.Config()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.settings.Config'
    assert var_9.py_version == 'py3'
    assert f'{type(var_9.force_to_top).__module__}.{type(var_9.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.force_to_top) == 0
    assert f'{type(var_9.skip).__module__}.{type(var_9.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.skip) == 19
    assert f'{type(var_9.extend_skip).__module__}.{type(var_9.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.extend_skip) == 0
    assert f'{type(var_9.skip_glob).__module__}.{type(var_9.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.skip_glob) == 0
    assert f'{type(var_9.extend_skip_glob).__module__}.{type(var_9.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.extend_skip_glob) == 0
    assert var_9.skip_gitignore is False
    assert var_9.line_length == 79
    assert var_9.wrap_length == 0
    assert var_9.line_ending == ''
    assert var_9.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_9.no_sections is False
    assert f'{type(var_9.known_future_library).__module__}.{type(var_9.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_future_library) == 1
    assert f'{type(var_9.known_third_party).__module__}.{type(var_9.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_third_party) == 0
    assert f'{type(var_9.known_first_party).__module__}.{type(var_9.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_first_party) == 0
    assert f'{type(var_9.known_local_folder).__module__}.{type(var_9.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_local_folder) == 0
    assert f'{type(var_9.known_standard_library).__module__}.{type(var_9.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_standard_library) == 234
    assert f'{type(var_9.extra_standard_library).__module__}.{type(var_9.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.extra_standard_library) == 0
    assert var_9.known_other == {}
    assert var_9.multi_line_output == module_5.WrapModes.GRID
    assert var_9.forced_separate == ()
    assert var_9.indent == '    '
    assert var_9.comment_prefix == '  #'
    assert var_9.length_sort is False
    assert var_9.length_sort_straight is False
    assert f'{type(var_9.length_sort_sections).__module__}.{type(var_9.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.length_sort_sections) == 0
    assert f'{type(var_9.add_imports).__module__}.{type(var_9.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.add_imports) == 0
    assert f'{type(var_9.remove_imports).__module__}.{type(var_9.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.remove_imports) == 0
    assert var_9.append_only is False
    assert var_9.reverse_relative is False
    assert var_9.force_single_line is False
    assert var_9.single_line_exclusions == ()
    assert var_9.default_section == 'THIRDPARTY'
    assert var_9.import_headings == {}
    assert var_9.import_footers == {}
    assert var_9.balanced_wrapping is False
    assert var_9.use_parentheses is False
    assert var_9.order_by_type is True
    assert var_9.atomic is False
    assert var_9.lines_before_imports == -1
    assert var_9.lines_after_imports == -1
    assert var_9.lines_between_sections == 1
    assert var_9.lines_between_types == 0
    assert var_9.combine_as_imports is False
    assert var_9.combine_star is False
    assert var_9.include_trailing_comma is False
    assert var_9.from_first is False
    assert var_9.verbose is False
    assert var_9.quiet is False
    assert var_9.force_adds is False
    assert var_9.force_alphabetical_sort_within_sections is False
    assert var_9.force_alphabetical_sort is False
    assert var_9.force_grid_wrap == 0
    assert var_9.force_sort_within_sections is False
    assert var_9.lexicographical is False
    assert var_9.group_by_package is False
    assert var_9.ignore_whitespace is False
    assert f'{type(var_9.no_lines_before).__module__}.{type(var_9.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.no_lines_before) == 0
    assert var_9.no_inline_sort is False
    assert var_9.ignore_comments is False
    assert var_9.case_sensitive is False
    assert f'{type(var_9.sources).__module__}.{type(var_9.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_9.sources) == 1
    assert var_9.virtual_env == ''
    assert var_9.conda_env == ''
    assert var_9.ensure_newline_before_comments is False
    assert var_9.directory == '/workspace'
    assert var_9.profile == ''
    assert var_9.honor_noqa is False
    assert f'{type(var_9.src_paths).__module__}.{type(var_9.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_9.src_paths) == 2
    assert var_9.remove_redundant_aliases is False
    assert var_9.float_to_top is False
    assert var_9.filter_files is False
    assert var_9.formatter == ''
    assert var_9.formatting_function is None
    assert var_9.color_output is False
    assert f'{type(var_9.treat_comments_as_code).__module__}.{type(var_9.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.treat_comments_as_code) == 0
    assert var_9.treat_all_comments_as_code is False
    assert f'{type(var_9.supported_extensions).__module__}.{type(var_9.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.supported_extensions) == 4
    assert f'{type(var_9.blocked_extensions).__module__}.{type(var_9.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.blocked_extensions) == 1
    assert f'{type(var_9.constants).__module__}.{type(var_9.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.constants) == 0
    assert f'{type(var_9.classes).__module__}.{type(var_9.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.classes) == 0
    assert f'{type(var_9.variables).__module__}.{type(var_9.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.variables) == 0
    assert var_9.dedup_headings is False
    assert var_9.only_sections is False
    assert var_9.only_modified is False
    assert var_9.combine_straight_imports is False
    assert var_9.auto_identify_namespace_packages is True
    assert f'{type(var_9.namespace_packages).__module__}.{type(var_9.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.namespace_packages) == 0
    assert var_9.follow_links is True
    assert var_9.indented_import_headings is True
    assert var_9.honor_case_in_force_sorted_sections is False
    assert var_9.sort_relative_in_force_sorted_sections is False
    assert var_9.overwrite_in_place is False
    assert var_9.reverse_sort is False
    assert var_9.star_first is False
    assert var_9.git_ls_files == {}
    assert var_9.format_error == '{error}: {message}'
    assert var_9.format_success == '{success}: {message}'
    assert var_9.sort_order == 'natural'
    assert var_9.sort_reexports is False
    assert var_9.split_on_trailing_comma is False
    var_10 = module_3.StringIO()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == '_io.StringIO'
    var_11 = module_4.Config()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.settings.Config'
    assert var_11.py_version == 'py3'
    assert f'{type(var_11.force_to_top).__module__}.{type(var_11.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.force_to_top) == 0
    assert f'{type(var_11.skip).__module__}.{type(var_11.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.skip) == 19
    assert f'{type(var_11.extend_skip).__module__}.{type(var_11.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.extend_skip) == 0
    assert f'{type(var_11.skip_glob).__module__}.{type(var_11.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.skip_glob) == 0
    assert f'{type(var_11.extend_skip_glob).__module__}.{type(var_11.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.extend_skip_glob) == 0
    assert var_11.skip_gitignore is False
    assert var_11.line_length == 79
    assert var_11.wrap_length == 0
    assert var_11.line_ending == ''
    assert var_11.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_11.no_sections is False
    assert f'{type(var_11.known_future_library).__module__}.{type(var_11.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_future_library) == 1
    assert f'{type(var_11.known_third_party).__module__}.{type(var_11.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_third_party) == 0
    assert f'{type(var_11.known_first_party).__module__}.{type(var_11.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_first_party) == 0
    assert f'{type(var_11.known_local_folder).__module__}.{type(var_11.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_local_folder) == 0
    assert f'{type(var_11.known_standard_library).__module__}.{type(var_11.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_standard_library) == 234
    assert f'{type(var_11.extra_standard_library).__module__}.{type(var_11.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.extra_standard_library) == 0
    assert var_11.known_other == {}
    assert var_11.multi_line_output == module_5.WrapModes.GRID
    assert var_11.forced_separate == ()
    assert var_11.indent == '    '
    assert var_11.comment_prefix == '  #'
    assert var_11.length_sort is False
    assert var_11.length_sort_straight is False
    assert f'{type(var_11.length_sort_sections).__module__}.{type(var_11.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.length_sort_sections) == 0
    assert f'{type(var_11.add_imports).__module__}.{type(var_11.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.add_imports) == 0
    assert f'{type(var_11.remove_imports).__module__}.{type(var_11.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.remove_imports) == 0
    assert var_11.append_only is False
    assert var_11.reverse_relative is False
    assert var_11.force_single_line is False
    assert var_11.single_line_exclusions == ()
    assert var_11.default_section == 'THIRDPARTY'
    assert var_11.import_headings == {}
    assert var_11.import_footers == {}
    assert var_11.balanced_wrapping is False
    assert var_11.use_parentheses is False
    assert var_11.order_by_type is True
    assert var_11.atomic is False
    assert var_11.lines_before_imports == -1
    assert var_11.lines_after_imports == -1
    assert var_11.lines_between_sections == 1
    assert var_11.lines_between_types == 0
    assert var_11.combine_as_imports is False
    assert var_11.combine_star is False
    assert var_11.include_trailing_comma is False
    assert var_11.from_first is False
    assert var_11.verbose is False
    assert var_11.quiet is False
    assert var_11.force_adds is False
    assert var_11.force_alphabetical_sort_within_sections is False
    assert var_11.force_alphabetical_sort is False
    assert var_11.force_grid_wrap == 0
    assert var_11.force_sort_within_sections is False
    assert var_11.lexicographical is False
    assert var_11.group_by_package is False
    assert var_11.ignore_whitespace is False
    assert f'{type(var_11.no_lines_before).__module__}.{type(var_11.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.no_lines_before) == 0
    assert var_11.no_inline_sort is False
    assert var_11.ignore_comments is False
    assert var_11.case_sensitive is False
    assert f'{type(var_11.sources).__module__}.{type(var_11.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_11.sources) == 1
    assert var_11.virtual_env == ''
    assert var_11.conda_env == ''
    assert var_11.ensure_newline_before_comments is False
    assert var_11.directory == '/workspace'
    assert var_11.profile == ''
    assert var_11.honor_noqa is False
    assert f'{type(var_11.src_paths).__module__}.{type(var_11.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_11.src_paths) == 2
    assert var_11.remove_redundant_aliases is False
    assert var_11.float_to_top is False
    assert var_11.filter_files is False
    assert var_11.formatter == ''
    assert var_11.formatting_function is None
    assert var_11.color_output is False
    assert f'{type(var_11.treat_comments_as_code).__module__}.{type(var_11.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.treat_comments_as_code) == 0
    assert var_11.treat_all_comments_as_code is False
    assert f'{type(var_11.supported_extensions).__module__}.{type(var_11.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.supported_extensions) == 4
    assert f'{type(var_11.blocked_extensions).__module__}.{type(var_11.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.blocked_extensions) == 1
    assert f'{type(var_11.constants).__module__}.{type(var_11.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.constants) == 0
    assert f'{type(var_11.classes).__module__}.{type(var_11.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.classes) == 0
    assert f'{type(var_11.variables).__module__}.{type(var_11.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.variables) == 0
    assert var_11.dedup_headings is False
    assert var_11.only_sections is False
    assert var_11.only_modified is False
    assert var_11.combine_straight_imports is False
    assert var_11.auto_identify_namespace_packages is True
    assert f'{type(var_11.namespace_packages).__module__}.{type(var_11.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.namespace_packages) == 0
    assert var_11.follow_links is True
    assert var_11.indented_import_headings is True
    assert var_11.honor_case_in_force_sorted_sections is False
    assert var_11.sort_relative_in_force_sorted_sections is False
    assert var_11.overwrite_in_place is False
    assert var_11.reverse_sort is False
    assert var_11.star_first is False
    assert var_11.git_ls_files == {}
    assert var_11.format_error == '{error}: {message}'
    assert var_11.format_success == '{success}: {message}'
    assert var_11.sort_order == 'natural'
    assert var_11.sort_reexports is False
    assert var_11.split_on_trailing_comma is False
    var_12 = module_3.StringIO()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == '_io.StringIO'
    var_13 = module_3.StringIO()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == '_io.StringIO'
    var_14 = module_4.Config()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'isort.settings.Config'
    assert var_14.py_version == 'py3'
    assert f'{type(var_14.force_to_top).__module__}.{type(var_14.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.force_to_top) == 0
    assert f'{type(var_14.skip).__module__}.{type(var_14.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.skip) == 19
    assert f'{type(var_14.extend_skip).__module__}.{type(var_14.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extend_skip) == 0
    assert f'{type(var_14.skip_glob).__module__}.{type(var_14.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.skip_glob) == 0
    assert f'{type(var_14.extend_skip_glob).__module__}.{type(var_14.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extend_skip_glob) == 0
    assert var_14.skip_gitignore is False
    assert var_14.line_length == 79
    assert var_14.wrap_length == 0
    assert var_14.line_ending == ''
    assert var_14.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_14.no_sections is False
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
    assert var_14.multi_line_output == module_5.WrapModes.GRID
    assert var_14.forced_separate == ()
    assert var_14.indent == '    '
    assert var_14.comment_prefix == '  #'
    assert var_14.length_sort is False
    assert var_14.length_sort_straight is False
    assert f'{type(var_14.length_sort_sections).__module__}.{type(var_14.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.length_sort_sections) == 0
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
    assert var_14.use_parentheses is False
    assert var_14.order_by_type is True
    assert var_14.atomic is False
    assert var_14.lines_before_imports == -1
    assert var_14.lines_after_imports == -1
    assert var_14.lines_between_sections == 1
    assert var_14.lines_between_types == 0
    assert var_14.combine_as_imports is False
    assert var_14.combine_star is False
    assert var_14.include_trailing_comma is False
    assert var_14.from_first is False
    assert var_14.verbose is False
    assert var_14.quiet is False
    assert var_14.force_adds is False
    assert var_14.force_alphabetical_sort_within_sections is False
    assert var_14.force_alphabetical_sort is False
    assert var_14.force_grid_wrap == 0
    assert var_14.force_sort_within_sections is False
    assert var_14.lexicographical is False
    assert var_14.group_by_package is False
    assert var_14.ignore_whitespace is False
    assert f'{type(var_14.no_lines_before).__module__}.{type(var_14.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.no_lines_before) == 0
    assert var_14.no_inline_sort is False
    assert var_14.ignore_comments is False
    assert var_14.case_sensitive is False
    assert f'{type(var_14.sources).__module__}.{type(var_14.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_14.sources) == 1
    assert var_14.virtual_env == ''
    assert var_14.conda_env == ''
    assert var_14.ensure_newline_before_comments is False
    assert var_14.directory == '/workspace'
    assert var_14.profile == ''
    assert var_14.honor_noqa is False
    assert f'{type(var_14.src_paths).__module__}.{type(var_14.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_14.src_paths) == 2
    assert var_14.remove_redundant_aliases is False
    assert var_14.float_to_top is False
    assert var_14.filter_files is False
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
    assert f'{type(var_14.constants).__module__}.{type(var_14.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.constants) == 0
    assert f'{type(var_14.classes).__module__}.{type(var_14.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.classes) == 0
    assert f'{type(var_14.variables).__module__}.{type(var_14.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.variables) == 0
    assert var_14.dedup_headings is False
    assert var_14.only_sections is False
    assert var_14.only_modified is False
    assert var_14.combine_straight_imports is False
    assert var_14.auto_identify_namespace_packages is True
    assert f'{type(var_14.namespace_packages).__module__}.{type(var_14.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.namespace_packages) == 0
    assert var_14.follow_links is True
    assert var_14.indented_import_headings is True
    assert var_14.honor_case_in_force_sorted_sections is False
    assert var_14.sort_relative_in_force_sorted_sections is False
    assert var_14.overwrite_in_place is False
    assert var_14.reverse_sort is False
    assert var_14.star_first is False
    assert var_14.git_ls_files == {}
    assert var_14.format_error == '{error}: {message}'
    assert var_14.format_success == '{success}: {message}'
    assert var_14.sort_order == 'natural'
    assert var_14.sort_reexports is False
    assert var_14.split_on_trailing_comma is False
    var_15 = module_3.StringIO()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == '_io.StringIO'
    var_16 = module_3.StringIO()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == '_io.StringIO'
    var_17 = '# First party'
    var_18 = [var_17]
    var_19 = module_4.Config()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'isort.settings.Config'
    assert var_19.py_version == 'py3'
    assert f'{type(var_19.force_to_top).__module__}.{type(var_19.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.force_to_top) == 0
    assert f'{type(var_19.skip).__module__}.{type(var_19.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.skip) == 19
    assert f'{type(var_19.extend_skip).__module__}.{type(var_19.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.extend_skip) == 0
    assert f'{type(var_19.skip_glob).__module__}.{type(var_19.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.skip_glob) == 0
    assert f'{type(var_19.extend_skip_glob).__module__}.{type(var_19.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.extend_skip_glob) == 0
    assert var_19.skip_gitignore is False
    assert var_19.line_length == 79
    assert var_19.wrap_length == 0
    assert var_19.line_ending == ''
    assert var_19.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_19.no_sections is False
    assert f'{type(var_19.known_future_library).__module__}.{type(var_19.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.known_future_library) == 1
    assert f'{type(var_19.known_third_party).__module__}.{type(var_19.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.known_third_party) == 0
    assert f'{type(var_19.known_first_party).__module__}.{type(var_19.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.known_first_party) == 0
    assert f'{type(var_19.known_local_folder).__module__}.{type(var_19.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.known_local_folder) == 0
    assert f'{type(var_19.known_standard_library).__module__}.{type(var_19.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.known_standard_library) == 234
    assert f'{type(var_19.extra_standard_library).__module__}.{type(var_19.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.extra_standard_library) == 0
    assert var_19.known_other == {}
    assert var_19.multi_line_output == module_5.WrapModes.GRID
    assert var_19.forced_separate == ()
    assert var_19.indent == '    '
    assert var_19.comment_prefix == '  #'
    assert var_19.length_sort is False
    assert var_19.length_sort_straight is False
    assert f'{type(var_19.length_sort_sections).__module__}.{type(var_19.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.length_sort_sections) == 0
    assert f'{type(var_19.add_imports).__module__}.{type(var_19.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.add_imports) == 0
    assert f'{type(var_19.remove_imports).__module__}.{type(var_19.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.remove_imports) == 0
    assert var_19.append_only is False
    assert var_19.reverse_relative is False
    assert var_19.force_single_line is False
    assert var_19.single_line_exclusions == ()
    assert var_19.default_section == 'THIRDPARTY'
    assert var_19.import_headings == {}
    assert var_19.import_footers == {}
    assert var_19.balanced_wrapping is False
    assert var_19.use_parentheses is False
    assert var_19.order_by_type is True
    assert var_19.atomic is False
    assert var_19.lines_before_imports == -1
    assert var_19.lines_after_imports == -1
    assert var_19.lines_between_sections == 1
    assert var_19.lines_between_types == 0
    assert var_19.combine_as_imports is False
    assert var_19.combine_star is False
    assert var_19.include_trailing_comma is False
    assert var_19.from_first is False
    assert var_19.verbose is False
    assert var_19.quiet is False
    assert var_19.force_adds is False
    assert var_19.force_alphabetical_sort_within_sections is False
    assert var_19.force_alphabetical_sort is False
    assert var_19.force_grid_wrap == 0
    assert var_19.force_sort_within_sections is False
    assert var_19.lexicographical is False
    assert var_19.group_by_package is False
    assert var_19.ignore_whitespace is False
    assert f'{type(var_19.no_lines_before).__module__}.{type(var_19.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.no_lines_before) == 0
    assert var_19.no_inline_sort is False
    assert var_19.ignore_comments is False
    assert var_19.case_sensitive is False
    assert f'{type(var_19.sources).__module__}.{type(var_19.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_19.sources) == 1
    assert var_19.virtual_env == ''
    assert var_19.conda_env == ''
    assert var_19.ensure_newline_before_comments is False
    assert var_19.directory == '/workspace'
    assert var_19.profile == ''
    assert var_19.honor_noqa is False
    assert f'{type(var_19.src_paths).__module__}.{type(var_19.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_19.src_paths) == 2
    assert var_19.remove_redundant_aliases is False
    assert var_19.float_to_top is False
    assert var_19.filter_files is False
    assert var_19.formatter == ''
    assert var_19.formatting_function is None
    assert var_19.color_output is False
    assert f'{type(var_19.treat_comments_as_code).__module__}.{type(var_19.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.treat_comments_as_code) == 0
    assert var_19.treat_all_comments_as_code is False
    assert f'{type(var_19.supported_extensions).__module__}.{type(var_19.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.supported_extensions) == 4
    assert f'{type(var_19.blocked_extensions).__module__}.{type(var_19.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.blocked_extensions) == 1
    assert f'{type(var_19.constants).__module__}.{type(var_19.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.constants) == 0
    assert f'{type(var_19.classes).__module__}.{type(var_19.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.classes) == 0
    assert f'{type(var_19.variables).__module__}.{type(var_19.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.variables) == 0
    assert var_19.dedup_headings is False
    assert var_19.only_sections is False
    assert var_19.only_modified is False
    assert var_19.combine_straight_imports is False
    assert var_19.auto_identify_namespace_packages is True
    assert f'{type(var_19.namespace_packages).__module__}.{type(var_19.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_19.namespace_packages) == 0
    assert var_19.follow_links is True
    assert var_19.indented_import_headings is True
    assert var_19.honor_case_in_force_sorted_sections is False
    assert var_19.sort_relative_in_force_sorted_sections is False
    assert var_19.overwrite_in_place is False
    assert var_19.reverse_sort is False
    assert var_19.star_first is False
    assert var_19.git_ls_files == {}
    assert var_19.format_error == '{error}: {message}'
    assert var_19.format_success == '{success}: {message}'
    assert var_19.sort_order == 'natural'
    assert var_19.sort_reexports is False
    assert var_19.split_on_trailing_comma is False
    var_20 = module_3.StringIO()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == '_io.StringIO'
    var_21 = module_3.StringIO()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == '_io.StringIO'
    var_22 = module_3.StringIO()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == '_io.StringIO'
    var_23 = module_4.Config()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'isort.settings.Config'
    assert var_23.py_version == 'py3'
    assert f'{type(var_23.force_to_top).__module__}.{type(var_23.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.force_to_top) == 0
    assert f'{type(var_23.skip).__module__}.{type(var_23.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.skip) == 19
    assert f'{type(var_23.extend_skip).__module__}.{type(var_23.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.extend_skip) == 0
    assert f'{type(var_23.skip_glob).__module__}.{type(var_23.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.skip_glob) == 0
    assert f'{type(var_23.extend_skip_glob).__module__}.{type(var_23.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.extend_skip_glob) == 0
    assert var_23.skip_gitignore is False
    assert var_23.line_length == 79
    assert var_23.wrap_length == 0
    assert var_23.line_ending == ''
    assert var_23.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_23.no_sections is False
    assert f'{type(var_23.known_future_library).__module__}.{type(var_23.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.known_future_library) == 1
    assert f'{type(var_23.known_third_party).__module__}.{type(var_23.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.known_third_party) == 0
    assert f'{type(var_23.known_first_party).__module__}.{type(var_23.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.known_first_party) == 0
    assert f'{type(var_23.known_local_folder).__module__}.{type(var_23.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.known_local_folder) == 0
    assert f'{type(var_23.known_standard_library).__module__}.{type(var_23.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.known_standard_library) == 234
    assert f'{type(var_23.extra_standard_library).__module__}.{type(var_23.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.extra_standard_library) == 0
    assert var_23.known_other == {}
    assert var_23.multi_line_output == module_5.WrapModes.GRID
    assert var_23.forced_separate == ()
    assert var_23.indent == '    '
    assert var_23.comment_prefix == '  #'
    assert var_23.length_sort is False
    assert var_23.length_sort_straight is False
    assert f'{type(var_23.length_sort_sections).__module__}.{type(var_23.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.length_sort_sections) == 0
    assert f'{type(var_23.add_imports).__module__}.{type(var_23.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.add_imports) == 0
    assert f'{type(var_23.remove_imports).__module__}.{type(var_23.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.remove_imports) == 0
    assert var_23.append_only is False
    assert var_23.reverse_relative is False
    assert var_23.force_single_line is False
    assert var_23.single_line_exclusions == ()
    assert var_23.default_section == 'THIRDPARTY'
    assert var_23.import_headings == {}
    assert var_23.import_footers == {}
    assert var_23.balanced_wrapping is False
    assert var_23.use_parentheses is False
    assert var_23.order_by_type is True
    assert var_23.atomic is False
    assert var_23.lines_before_imports == -1
    assert var_23.lines_after_imports == -1
    assert var_23.lines_between_sections == 1
    assert var_23.lines_between_types == 0
    assert var_23.combine_as_imports is False
    assert var_23.combine_star is False
    assert var_23.include_trailing_comma is False
    assert var_23.from_first is False
    assert var_23.verbose is False
    assert var_23.quiet is False
    assert var_23.force_adds is False
    assert var_23.force_alphabetical_sort_within_sections is False
    assert var_23.force_alphabetical_sort is False
    assert var_23.force_grid_wrap == 0
    assert var_23.force_sort_within_sections is False
    assert var_23.lexicographical is False
    assert var_23.group_by_package is False
    assert var_23.ignore_whitespace is False
    assert f'{type(var_23.no_lines_before).__module__}.{type(var_23.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.no_lines_before) == 0
    assert var_23.no_inline_sort is False
    assert var_23.ignore_comments is False
    assert var_23.case_sensitive is False
    assert f'{type(var_23.sources).__module__}.{type(var_23.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_23.sources) == 1
    assert var_23.virtual_env == ''
    assert var_23.conda_env == ''
    assert var_23.ensure_newline_before_comments is False
    assert var_23.directory == '/workspace'
    assert var_23.profile == ''
    assert var_23.honor_noqa is False
    assert f'{type(var_23.src_paths).__module__}.{type(var_23.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_23.src_paths) == 2
    assert var_23.remove_redundant_aliases is False
    assert var_23.float_to_top is False
    assert var_23.filter_files is False
    assert var_23.formatter == ''
    assert var_23.formatting_function is None
    assert var_23.color_output is False
    assert f'{type(var_23.treat_comments_as_code).__module__}.{type(var_23.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.treat_comments_as_code) == 0
    assert var_23.treat_all_comments_as_code is False
    assert f'{type(var_23.supported_extensions).__module__}.{type(var_23.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.supported_extensions) == 4
    assert f'{type(var_23.blocked_extensions).__module__}.{type(var_23.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.blocked_extensions) == 1
    assert f'{type(var_23.constants).__module__}.{type(var_23.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.constants) == 0
    assert f'{type(var_23.classes).__module__}.{type(var_23.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.classes) == 0
    assert f'{type(var_23.variables).__module__}.{type(var_23.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.variables) == 0
    assert var_23.dedup_headings is False
    assert var_23.only_sections is False
    assert var_23.only_modified is False
    assert var_23.combine_straight_imports is False
    assert var_23.auto_identify_namespace_packages is True
    assert f'{type(var_23.namespace_packages).__module__}.{type(var_23.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_23.namespace_packages) == 0
    assert var_23.follow_links is True
    assert var_23.indented_import_headings is True
    assert var_23.honor_case_in_force_sorted_sections is False
    assert var_23.sort_relative_in_force_sorted_sections is False
    assert var_23.overwrite_in_place is False
    assert var_23.reverse_sort is False
    assert var_23.star_first is False
    assert var_23.git_ls_files == {}
    assert var_23.format_error == '{error}: {message}'
    assert var_23.format_success == '{success}: {message}'
    assert var_23.sort_order == 'natural'
    assert var_23.sort_reexports is False
    assert var_23.split_on_trailing_comma is False
    var_24 = module_3.StringIO()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == '_io.StringIO'
    var_25 = module_4.Config()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'isort.settings.Config'
    assert var_25.py_version == 'py3'
    assert f'{type(var_25.force_to_top).__module__}.{type(var_25.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.force_to_top) == 0
    assert f'{type(var_25.skip).__module__}.{type(var_25.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.skip) == 19
    assert f'{type(var_25.extend_skip).__module__}.{type(var_25.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.extend_skip) == 0
    assert f'{type(var_25.skip_glob).__module__}.{type(var_25.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.skip_glob) == 0
    assert f'{type(var_25.extend_skip_glob).__module__}.{type(var_25.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.extend_skip_glob) == 0
    assert var_25.skip_gitignore is False
    assert var_25.line_length == 79
    assert var_25.wrap_length == 0
    assert var_25.line_ending == ''
    assert var_25.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_25.no_sections is False
    assert f'{type(var_25.known_future_library).__module__}.{type(var_25.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.known_future_library) == 1
    assert f'{type(var_25.known_third_party).__module__}.{type(var_25.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.known_third_party) == 0
    assert f'{type(var_25.known_first_party).__module__}.{type(var_25.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.known_first_party) == 0
    assert f'{type(var_25.known_local_folder).__module__}.{type(var_25.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.known_local_folder) == 0
    assert f'{type(var_25.known_standard_library).__module__}.{type(var_25.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.known_standard_library) == 234
    assert f'{type(var_25.extra_standard_library).__module__}.{type(var_25.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.extra_standard_library) == 0
    assert var_25.known_other == {}
    assert var_25.multi_line_output == module_5.WrapModes.GRID
    assert var_25.forced_separate == ()
    assert var_25.indent == '    '
    assert var_25.comment_prefix == '  #'
    assert var_25.length_sort is False
    assert var_25.length_sort_straight is False
    assert f'{type(var_25.length_sort_sections).__module__}.{type(var_25.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.length_sort_sections) == 0
    assert f'{type(var_25.add_imports).__module__}.{type(var_25.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.add_imports) == 0
    assert f'{type(var_25.remove_imports).__module__}.{type(var_25.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.remove_imports) == 0
    assert var_25.append_only is False
    assert var_25.reverse_relative is False
    assert var_25.force_single_line is False
    assert var_25.single_line_exclusions == ()
    assert var_25.default_section == 'THIRDPARTY'
    assert var_25.import_headings == {}
    assert var_25.import_footers == {}
    assert var_25.balanced_wrapping is False
    assert var_25.use_parentheses is False
    assert var_25.order_by_type is True
    assert var_25.atomic is False
    assert var_25.lines_before_imports == -1
    assert var_25.lines_after_imports == -1
    assert var_25.lines_between_sections == 1
    assert var_25.lines_between_types == 0
    assert var_25.combine_as_imports is False
    assert var_25.combine_star is False
    assert var_25.include_trailing_comma is False
    assert var_25.from_first is False
    assert var_25.verbose is False
    assert var_25.quiet is False
    assert var_25.force_adds is False
    assert var_25.force_alphabetical_sort_within_sections is False
    assert var_25.force_alphabetical_sort is False
    assert var_25.force_grid_wrap == 0
    assert var_25.force_sort_within_sections is False
    assert var_25.lexicographical is False
    assert var_25.group_by_package is False
    assert var_25.ignore_whitespace is False
    assert f'{type(var_25.no_lines_before).__module__}.{type(var_25.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.no_lines_before) == 0
    assert var_25.no_inline_sort is False
    assert var_25.ignore_comments is False
    assert var_25.case_sensitive is False
    assert f'{type(var_25.sources).__module__}.{type(var_25.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_25.sources) == 1
    assert var_25.virtual_env == ''
    assert var_25.conda_env == ''
    assert var_25.ensure_newline_before_comments is False
    assert var_25.directory == '/workspace'
    assert var_25.profile == ''
    assert var_25.honor_noqa is False
    assert f'{type(var_25.src_paths).__module__}.{type(var_25.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_25.src_paths) == 2
    assert var_25.remove_redundant_aliases is False
    assert var_25.float_to_top is False
    assert var_25.filter_files is False
    assert var_25.formatter == ''
    assert var_25.formatting_function is None
    assert var_25.color_output is False
    assert f'{type(var_25.treat_comments_as_code).__module__}.{type(var_25.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.treat_comments_as_code) == 0
    assert var_25.treat_all_comments_as_code is False
    assert f'{type(var_25.supported_extensions).__module__}.{type(var_25.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.supported_extensions) == 4
    assert f'{type(var_25.blocked_extensions).__module__}.{type(var_25.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.blocked_extensions) == 1
    assert f'{type(var_25.constants).__module__}.{type(var_25.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.constants) == 0
    assert f'{type(var_25.classes).__module__}.{type(var_25.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.classes) == 0
    assert f'{type(var_25.variables).__module__}.{type(var_25.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.variables) == 0
    assert var_25.dedup_headings is False
    assert var_25.only_sections is False
    assert var_25.only_modified is False
    assert var_25.combine_straight_imports is False
    assert var_25.auto_identify_namespace_packages is True
    assert f'{type(var_25.namespace_packages).__module__}.{type(var_25.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_25.namespace_packages) == 0
    assert var_25.follow_links is True
    assert var_25.indented_import_headings is True
    assert var_25.honor_case_in_force_sorted_sections is False
    assert var_25.sort_relative_in_force_sorted_sections is False
    assert var_25.overwrite_in_place is False
    assert var_25.reverse_sort is False
    assert var_25.star_first is False
    assert var_25.git_ls_files == {}
    assert var_25.format_error == '{error}: {message}'
    assert var_25.format_success == '{success}: {message}'
    assert var_25.sort_order == 'natural'
    assert var_25.sort_reexports is False
    assert var_25.split_on_trailing_comma is False
    var_26 = module_3.StringIO()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == '_io.StringIO'
    var_27 = module_4.Config()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'isort.settings.Config'
    assert var_27.py_version == 'py3'
    assert f'{type(var_27.force_to_top).__module__}.{type(var_27.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.force_to_top) == 0
    assert f'{type(var_27.skip).__module__}.{type(var_27.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.skip) == 19
    assert f'{type(var_27.extend_skip).__module__}.{type(var_27.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.extend_skip) == 0
    assert f'{type(var_27.skip_glob).__module__}.{type(var_27.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.skip_glob) == 0
    assert f'{type(var_27.extend_skip_glob).__module__}.{type(var_27.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.extend_skip_glob) == 0
    assert var_27.skip_gitignore is False
    assert var_27.line_length == 79
    assert var_27.wrap_length == 0
    assert var_27.line_ending == ''
    assert var_27.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_27.no_sections is False
    assert f'{type(var_27.known_future_library).__module__}.{type(var_27.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_future_library) == 1
    assert f'{type(var_27.known_third_party).__module__}.{type(var_27.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_third_party) == 0
    assert f'{type(var_27.known_first_party).__module__}.{type(var_27.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_first_party) == 0
    assert f'{type(var_27.known_local_folder).__module__}.{type(var_27.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_local_folder) == 0
    assert f'{type(var_27.known_standard_library).__module__}.{type(var_27.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_standard_library) == 234
    assert f'{type(var_27.extra_standard_library).__module__}.{type(var_27.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.extra_standard_library) == 0
    assert var_27.known_other == {}
    assert var_27.multi_line_output == module_5.WrapModes.GRID
    assert var_27.forced_separate == ()
    assert var_27.indent == '    '
    assert var_27.comment_prefix == '  #'
    assert var_27.length_sort is False
    assert var_27.length_sort_straight is False
    assert f'{type(var_27.length_sort_sections).__module__}.{type(var_27.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.length_sort_sections) == 0
    assert f'{type(var_27.add_imports).__module__}.{type(var_27.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.add_imports) == 0
    assert f'{type(var_27.remove_imports).__module__}.{type(var_27.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.remove_imports) == 0
    assert var_27.append_only is False
    assert var_27.reverse_relative is False
    assert var_27.force_single_line is False
    assert var_27.single_line_exclusions == ()
    assert var_27.default_section == 'THIRDPARTY'
    assert var_27.import_headings == {}
    assert var_27.import_footers == {}
    assert var_27.balanced_wrapping is False
    assert var_27.use_parentheses is False
    assert var_27.order_by_type is True
    assert var_27.atomic is False
    assert var_27.lines_before_imports == -1
    assert var_27.lines_after_imports == -1
    assert var_27.lines_between_sections == 1
    assert var_27.lines_between_types == 0
    assert var_27.combine_as_imports is False
    assert var_27.combine_star is False
    assert var_27.include_trailing_comma is False
    assert var_27.from_first is False
    assert var_27.verbose is False
    assert var_27.quiet is False
    assert var_27.force_adds is False
    assert var_27.force_alphabetical_sort_within_sections is False
    assert var_27.force_alphabetical_sort is False
    assert var_27.force_grid_wrap == 0
    assert var_27.force_sort_within_sections is False
    assert var_27.lexicographical is False
    assert var_27.group_by_package is False
    assert var_27.ignore_whitespace is False
    assert f'{type(var_27.no_lines_before).__module__}.{type(var_27.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.no_lines_before) == 0
    assert var_27.no_inline_sort is False
    assert var_27.ignore_comments is False
    assert var_27.case_sensitive is False
    assert f'{type(var_27.sources).__module__}.{type(var_27.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_27.sources) == 1
    assert var_27.virtual_env == ''
    assert var_27.conda_env == ''
    assert var_27.ensure_newline_before_comments is False
    assert var_27.directory == '/workspace'
    assert var_27.profile == ''
    assert var_27.honor_noqa is False
    assert f'{type(var_27.src_paths).__module__}.{type(var_27.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_27.src_paths) == 2
    assert var_27.remove_redundant_aliases is False
    assert var_27.float_to_top is False
    assert var_27.filter_files is False
    assert var_27.formatter == ''
    assert var_27.formatting_function is None
    assert var_27.color_output is False
    assert f'{type(var_27.treat_comments_as_code).__module__}.{type(var_27.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.treat_comments_as_code) == 0
    assert var_27.treat_all_comments_as_code is False
    assert f'{type(var_27.supported_extensions).__module__}.{type(var_27.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.supported_extensions) == 4
    assert f'{type(var_27.blocked_extensions).__module__}.{type(var_27.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.blocked_extensions) == 1
    assert f'{type(var_27.constants).__module__}.{type(var_27.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.constants) == 0
    assert f'{type(var_27.classes).__module__}.{type(var_27.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.classes) == 0
    assert f'{type(var_27.variables).__module__}.{type(var_27.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.variables) == 0
    assert var_27.dedup_headings is False
    assert var_27.only_sections is False
    assert var_27.only_modified is False
    assert var_27.combine_straight_imports is False
    assert var_27.auto_identify_namespace_packages is True
    assert f'{type(var_27.namespace_packages).__module__}.{type(var_27.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.namespace_packages) == 0
    assert var_27.follow_links is True
    assert var_27.indented_import_headings is True
    assert var_27.honor_case_in_force_sorted_sections is False
    assert var_27.sort_relative_in_force_sorted_sections is False
    assert var_27.overwrite_in_place is False
    assert var_27.reverse_sort is False
    assert var_27.star_first is False
    assert var_27.git_ls_files == {}
    assert var_27.format_error == '{error}: {message}'
    assert var_27.format_success == '{success}: {message}'
    assert var_27.sort_order == 'natural'
    assert var_27.sort_reexports is False
    assert var_27.split_on_trailing_comma is False
    var_28 = module_3.StringIO()
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == '_io.StringIO'
    var_29 = module_3.StringIO()
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == '_io.StringIO'
    var_30 = 'import test'
    var_31 = [var_30]
    var_32 = module_4.Config()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'isort.settings.Config'
    assert var_32.py_version == 'py3'
    assert f'{type(var_32.force_to_top).__module__}.{type(var_32.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.force_to_top) == 0
    assert f'{type(var_32.skip).__module__}.{type(var_32.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.skip) == 19
    assert f'{type(var_32.extend_skip).__module__}.{type(var_32.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.extend_skip) == 0
    assert f'{type(var_32.skip_glob).__module__}.{type(var_32.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.skip_glob) == 0
    assert f'{type(var_32.extend_skip_glob).__module__}.{type(var_32.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.extend_skip_glob) == 0
    assert var_32.skip_gitignore is False
    assert var_32.line_length == 79
    assert var_32.wrap_length == 0
    assert var_32.line_ending == ''
    assert var_32.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_32.no_sections is False
    assert f'{type(var_32.known_future_library).__module__}.{type(var_32.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_future_library) == 1
    assert f'{type(var_32.known_third_party).__module__}.{type(var_32.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_third_party) == 0
    assert f'{type(var_32.known_first_party).__module__}.{type(var_32.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_first_party) == 0
    assert f'{type(var_32.known_local_folder).__module__}.{type(var_32.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_local_folder) == 0
    assert f'{type(var_32.known_standard_library).__module__}.{type(var_32.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_standard_library) == 234
    assert f'{type(var_32.extra_standard_library).__module__}.{type(var_32.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.extra_standard_library) == 0
    assert var_32.known_other == {}
    assert var_32.multi_line_output == module_5.WrapModes.GRID
    assert var_32.forced_separate == ()
    assert var_32.indent == '    '
    assert var_32.comment_prefix == '  #'
    assert var_32.length_sort is False
    assert var_32.length_sort_straight is False
    assert f'{type(var_32.length_sort_sections).__module__}.{type(var_32.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.length_sort_sections) == 0
    assert f'{type(var_32.add_imports).__module__}.{type(var_32.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.add_imports) == 0
    assert f'{type(var_32.remove_imports).__module__}.{type(var_32.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.remove_imports) == 0
    assert var_32.append_only is False
    assert var_32.reverse_relative is False
    assert var_32.force_single_line is False
    assert var_32.single_line_exclusions == ()
    assert var_32.default_section == 'THIRDPARTY'
    assert var_32.import_headings == {}
    assert var_32.import_footers == {}
    assert var_32.balanced_wrapping is False
    assert var_32.use_parentheses is False
    assert var_32.order_by_type is True
    assert var_32.atomic is False
    assert var_32.lines_before_imports == -1
    assert var_32.lines_after_imports == -1
    assert var_32.lines_between_sections == 1
    assert var_32.lines_between_types == 0
    assert var_32.combine_as_imports is False
    assert var_32.combine_star is False
    assert var_32.include_trailing_comma is False
    assert var_32.from_first is False
    assert var_32.verbose is False
    assert var_32.quiet is False
    assert var_32.force_adds is False
    assert var_32.force_alphabetical_sort_within_sections is False
    assert var_32.force_alphabetical_sort is False
    assert var_32.force_grid_wrap == 0
    assert var_32.force_sort_within_sections is False
    assert var_32.lexicographical is False
    assert var_32.group_by_package is False
    assert var_32.ignore_whitespace is False
    assert f'{type(var_32.no_lines_before).__module__}.{type(var_32.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.no_lines_before) == 0
    assert var_32.no_inline_sort is False
    assert var_32.ignore_comments is False
    assert var_32.case_sensitive is False
    assert f'{type(var_32.sources).__module__}.{type(var_32.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_32.sources) == 1
    assert var_32.virtual_env == ''
    assert var_32.conda_env == ''
    assert var_32.ensure_newline_before_comments is False
    assert var_32.directory == '/workspace'
    assert var_32.profile == ''
    assert var_32.honor_noqa is False
    assert f'{type(var_32.src_paths).__module__}.{type(var_32.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_32.src_paths) == 2
    assert var_32.remove_redundant_aliases is False
    assert var_32.float_to_top is False
    assert var_32.filter_files is False
    assert var_32.formatter == ''
    assert var_32.formatting_function is None
    assert var_32.color_output is False
    assert f'{type(var_32.treat_comments_as_code).__module__}.{type(var_32.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.treat_comments_as_code) == 0
    assert var_32.treat_all_comments_as_code is False
    assert f'{type(var_32.supported_extensions).__module__}.{type(var_32.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.supported_extensions) == 4
    assert f'{type(var_32.blocked_extensions).__module__}.{type(var_32.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.blocked_extensions) == 1
    assert f'{type(var_32.constants).__module__}.{type(var_32.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.constants) == 0
    assert f'{type(var_32.classes).__module__}.{type(var_32.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.classes) == 0
    assert f'{type(var_32.variables).__module__}.{type(var_32.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.variables) == 0
    assert var_32.dedup_headings is False
    assert var_32.only_sections is False
    assert var_32.only_modified is False
    assert var_32.combine_straight_imports is False
    assert var_32.auto_identify_namespace_packages is True
    assert f'{type(var_32.namespace_packages).__module__}.{type(var_32.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.namespace_packages) == 0
    assert var_32.follow_links is True
    assert var_32.indented_import_headings is True
    assert var_32.honor_case_in_force_sorted_sections is False
    assert var_32.sort_relative_in_force_sorted_sections is False
    assert var_32.overwrite_in_place is False
    assert var_32.reverse_sort is False
    assert var_32.star_first is False
    assert var_32.git_ls_files == {}
    assert var_32.format_error == '{error}: {message}'
    assert var_32.format_success == '{success}: {message}'
    assert var_32.sort_order == 'natural'
    assert var_32.sort_reexports is False
    assert var_32.split_on_trailing_comma is False
    var_33 = module_3.StringIO()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == '_io.StringIO'
    var_34 = module_3.StringIO()
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == '_io.StringIO'
    var_35 = 'a\nb'
    var_36 = '\n'
    var_37 = False
    var_38 = module_0._has_changed(var_35, var_35, var_36, var_37)
    assert var_38 is False
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_5.WrapModes.GRID
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
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.CIMPORT_IDENTIFIERS == ('cimport ', 'cimport*', 'from.cimport')
    assert module_0.IMPORT_START_IDENTIFIERS == ('from ', 'from.import', 'import ', 'import*', 'cimport ', 'cimport*', 'from.cimport')
    assert module_0.DOCSTRING_INDICATORS == ('"""', "'''")
    assert module_0.COMMENT_INDICATORS == ('"""', "'''", "'", '"', '#')
    assert module_0.CODE_SORT_COMMENTS == ('# isort: list', '# isort: dict', '# isort: set', '# isort: unique-list', '# isort: tuple', '# isort: unique-tuple', '# isort: assignments')
    assert module_0.LITERAL_TYPE_MAPPING == {'(': 'tuple', '[': 'list', '{': 'set'}
    var_39 = 'b\na'
    var_40 = module_0._has_changed(var_35, var_39, var_36, var_37)
    assert var_40 is True