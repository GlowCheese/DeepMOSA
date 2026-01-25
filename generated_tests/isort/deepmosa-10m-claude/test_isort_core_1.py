# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.core as module_0
import urllib.request as module_1
import isort.wrap_modes as module_2
import email._header_value_parser as module_3
import posixpath as module_4
import urllib.parse as module_5
import isort.settings as module_6

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.process(var_0, var_0)

def test_case_1():
    var_0 = module_1.noheaders()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'email.message.Message'
    assert len(var_0) == 0
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    var_1 = module_0.process(var_0, var_0)
    assert var_1 is False
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
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

def test_case_2():
    pass

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = ' ;q(s0)tX_BHwb'
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'from sy import argv\nfrom os import path\n'
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'Qz]=\x0b"LYQq$Bs!jp'
    var_1 = [var_0, var_0, var_0]
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_3.quote_string(var_0)
    assert var_1 == '"None"'
    assert module_3.hexdigits == '0123456789abcdefABCDEF'
    assert module_3.WSP == {' ', '\t'}
    assert module_3.CFWS_LEADER == {' ', '\t', '('}
    assert module_3.SPECIALS == {'"', '@', ':', '<', ';', '\\', '[', ']', '(', ',', ')', '>', '.'}
    assert module_3.ATOM_ENDS == {' ', '"', '@', ':', '\t', '<', ';', '\\', '[', ']', '(', ',', ')', '>', '.'}
    assert module_3.DOT_ATOM_ENDS == {' ', '"', '@', ':', '\t', '<', ';', '\\', '[', ']', '(', ',', ')', '>'}
    assert module_3.PHRASE_ENDS == {'@', ':', '<', ';', '\\', '[', ']', ',', ')', '>'}
    assert module_3.TSPECIALS == {'"', '@', ':', '[', '\\', '<', ';', ']', '(', '/', ',', ')', '>', '?', '='}
    assert module_3.TOKEN_ENDS == {' ', '"', '@', ':', '\t', '<', ';', '\\', '[', ']', '(', '/', ',', ')', '>', '?', '='}
    assert module_3.ASPECIALS == {'"', '@', ':', '%', '*', '<', ';', '\\', '[', ']', "'", '(', '/', ',', ')', '>', '?', '='}
    assert module_3.ATTRIBUTE_ENDS == {'*', '[', ';', ')', '"', '%', "'", ']', ',', '>', ' ', '@', '<', '\\', '?', ':', '\t', '(', '/', '='}
    assert module_3.EXTENDED_ATTRIBUTE_ENDS == {'*', '[', ';', ')', '"', "'", ']', ',', '>', ' ', '@', '<', '\\', '?', ':', '\t', '(', '/', '='}
    assert module_3.NLSET == {'\n', '\r'}
    assert module_3.SPECIALSNL == {'\n', '"', '@', '\r', ':', '<', ';', '\\', '[', ']', '(', ',', ')', '>', '.'}
    assert f'{type(module_3.rfc2047_matcher).__module__}.{type(module_3.rfc2047_matcher).__qualname__}' == 're.Pattern'
    assert f'{type(module_3.DOT).__module__}.{type(module_3.DOT).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_3.DOT) == 1
    assert f'{type(module_3.ListSeparator).__module__}.{type(module_3.ListSeparator).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_3.ListSeparator) == 1
    assert f'{type(module_3.RouteComponentMarker).__module__}.{type(module_3.RouteComponentMarker).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_3.RouteComponentMarker) == 1
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = "2'C$e7T"
    var_1 = module_4.splitext(var_0)
    assert module_4.curdir == '.'
    assert module_4.pardir == '..'
    assert module_4.extsep == '.'
    assert module_4.sep == '/'
    assert module_4.pathsep == ':'
    assert module_4.defpath == '/bin:/usr/bin'
    assert module_4.altsep is None
    assert module_4.devnull == '/dev/null'
    assert f'{type(module_4.ALLOW_MISSING).__module__}.{type(module_4.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    assert module_4.supports_unicode_filenames is False
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '# isort: skip_file\nimport sys\n'
    var_1 = None
    module_0.process(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'from sys import argv\nfrom os import path\n'
    var_1 = [var_0, var_0]
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'import s'
    var_1 = module_5.urldefrag(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'urllib.parse.DefragResult'
    assert len(var_1) == 2
    assert module_5.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_5.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_5.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_5.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_5.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_5.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_5.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_5.MAX_CACHE_SIZE == 20
    var_2 = 'rr&H#G37\rKX 9V@yO6'
    module_0.process(var_1, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'from sys import argvhf>om os import path\n'
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = ":zxm'dka'J!.&VNv\\"
    var_1 = [var_0]
    var_2 = None
    module_0.process(var_1, var_2, var_0, var_2)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'import json'
    var_1 = [var_0]
    var_2 = 'add_imports'
    var_3 = {var_2: var_1}
    var_4 = module_6.Config(**var_3)
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
    assert var_4.multi_line_output == module_2.WrapModes.GRID
    assert var_4.forced_separate == ()
    assert var_4.indent == '    '
    assert var_4.comment_prefix == '  #'
    assert var_4.length_sort is False
    assert var_4.length_sort_straight is False
    assert f'{type(var_4.length_sort_sections).__module__}.{type(var_4.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.length_sort_sections) == 0
    assert f'{type(var_4.add_imports).__module__}.{type(var_4.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.add_imports) == 1
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
    assert module_6.TYPE_CHECKING is False
    assert module_6.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_6.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_6.FIRSTPARTY == 'FIRSTPARTY'
    assert module_6.FUTURE == 'FUTURE'
    assert module_6.LOCALFOLDER == 'LOCALFOLDER'
    assert module_6.STDLIB == 'STDLIB'
    assert module_6.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_6.CYTHON_EXTENSIONS).__module__}.{type(module_6.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_6.SUPPORTED_EXTENSIONS).__module__}.{type(module_6.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_6.BLOCKED_EXTENSIONS).__module__}.{type(module_6.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.BLOCKED_EXTENSIONS) == 1
    assert module_6.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_6.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_6.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_6.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_6.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_6.DEFAULT_SKIP).__module__}.{type(module_6.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_SKIP) == 19
    assert module_6.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_6.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_6.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_6.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_6.KNOWN_PREFIX == 'known_'
    assert module_6.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_6.RUNTIME_SOURCE == 'runtime'
    assert module_6.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_6.DEFAULT_CONFIG).__module__}.{type(module_6.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_6.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_6.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_6.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.skip).__module__}.{type(module_6.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_6.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_6.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_6.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_6.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_6.DEFAULT_CONFIG.skip_gitignore is False
    assert module_6.DEFAULT_CONFIG.line_length == 79
    assert module_6.DEFAULT_CONFIG.wrap_length == 0
    assert module_6.DEFAULT_CONFIG.line_ending == ''
    assert module_6.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_6.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_6.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_6.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_6.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_6.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_6.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_6.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_6.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_6.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_6.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_6.DEFAULT_CONFIG.known_other == {}
    assert module_6.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_6.DEFAULT_CONFIG.forced_separate == ()
    assert module_6.DEFAULT_CONFIG.indent == '    '
    assert module_6.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_6.DEFAULT_CONFIG.length_sort is False
    assert module_6.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_6.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_6.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.add_imports).__module__}.{type(module_6.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_6.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.remove_imports) == 0
    assert module_6.DEFAULT_CONFIG.append_only is False
    assert module_6.DEFAULT_CONFIG.reverse_relative is False
    assert module_6.DEFAULT_CONFIG.force_single_line is False
    assert module_6.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_6.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_6.DEFAULT_CONFIG.import_headings == {}
    assert module_6.DEFAULT_CONFIG.import_footers == {}
    assert module_6.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_6.DEFAULT_CONFIG.use_parentheses is False
    assert module_6.DEFAULT_CONFIG.order_by_type is True
    assert module_6.DEFAULT_CONFIG.atomic is False
    assert module_6.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_6.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_6.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_6.DEFAULT_CONFIG.lines_between_types == 0
    assert module_6.DEFAULT_CONFIG.combine_as_imports is False
    assert module_6.DEFAULT_CONFIG.combine_star is False
    assert module_6.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_6.DEFAULT_CONFIG.from_first is False
    assert module_6.DEFAULT_CONFIG.verbose is False
    assert module_6.DEFAULT_CONFIG.quiet is False
    assert module_6.DEFAULT_CONFIG.force_adds is False
    assert module_6.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_6.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_6.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_6.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_6.DEFAULT_CONFIG.lexicographical is False
    assert module_6.DEFAULT_CONFIG.group_by_package is False
    assert module_6.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_6.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_6.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_6.DEFAULT_CONFIG.no_inline_sort is False
    assert module_6.DEFAULT_CONFIG.ignore_comments is False
    assert module_6.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_6.DEFAULT_CONFIG.sources).__module__}.{type(module_6.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_6.DEFAULT_CONFIG.sources) == 1
    assert module_6.DEFAULT_CONFIG.virtual_env == ''
    assert module_6.DEFAULT_CONFIG.conda_env == ''
    assert module_6.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_6.DEFAULT_CONFIG.directory == '/workspace'
    assert module_6.DEFAULT_CONFIG.profile == ''
    assert module_6.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_6.DEFAULT_CONFIG.src_paths).__module__}.{type(module_6.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_6.DEFAULT_CONFIG.src_paths) == 2
    assert module_6.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_6.DEFAULT_CONFIG.float_to_top is False
    assert module_6.DEFAULT_CONFIG.filter_files is False
    assert module_6.DEFAULT_CONFIG.formatter == ''
    assert module_6.DEFAULT_CONFIG.formatting_function is None
    assert module_6.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_6.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_6.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_6.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_6.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_6.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_6.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_6.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_6.DEFAULT_CONFIG.constants).__module__}.{type(module_6.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.classes).__module__}.{type(module_6.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.variables).__module__}.{type(module_6.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.variables) == 0
    assert module_6.DEFAULT_CONFIG.dedup_headings is False
    assert module_6.DEFAULT_CONFIG.only_sections is False
    assert module_6.DEFAULT_CONFIG.only_modified is False
    assert module_6.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_6.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_6.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_6.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_6.DEFAULT_CONFIG.follow_links is True
    assert module_6.DEFAULT_CONFIG.indented_import_headings is True
    assert module_6.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_6.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_6.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_6.DEFAULT_CONFIG.reverse_sort is False
    assert module_6.DEFAULT_CONFIG.star_first is False
    assert module_6.DEFAULT_CONFIG.git_ls_files == {}
    assert module_6.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_6.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_6.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_6.DEFAULT_CONFIG.sort_reexports is False
    assert module_6.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_6.Config.known_patterns).__module__}.{type(module_6.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.section_comments).__module__}.{type(module_6.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.section_comments_end).__module__}.{type(module_6.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.skips).__module__}.{type(module_6.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.skip_globs).__module__}.{type(module_6.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.sorting_function).__module__}.{type(module_6.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_5 = 'import os\n'
    var_6 = None
    var_7 = False
    module_0.process(var_6, var_6, var_5, var_7, var_4)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'from <y import argv\nfrom os import pathV'
    var_1 = module_4.splitext(var_0)
    assert module_4.curdir == '.'
    assert module_4.pardir == '..'
    assert module_4.extsep == '.'
    assert module_4.sep == '/'
    assert module_4.pathsep == ':'
    assert module_4.defpath == '/bin:/usr/bin'
    assert module_4.altsep is None
    assert module_4.devnull == '/dev/null'
    assert f'{type(module_4.ALLOW_MISSING).__module__}.{type(module_4.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    assert module_4.supports_unicode_filenames is False
    module_0.process(var_1, var_1, raise_on_skip=var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'from <y impor argv\nfrom os imort path\n'
    var_1 = module_4.splitext(var_0)
    assert module_4.curdir == '.'
    assert module_4.pardir == '..'
    assert module_4.extsep == '.'
    assert module_4.sep == '/'
    assert module_4.pathsep == ':'
    assert module_4.defpath == '/bin:/usr/bin'
    assert module_4.altsep is None
    assert module_4.devnull == '/dev/null'
    assert f'{type(module_4.ALLOW_MISSING).__module__}.{type(module_4.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    assert module_4.supports_unicode_filenames is False
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'from sy( iport argv\nOrom os import path&'
    var_1 = None
    var_2 = module_3.parse_message_id(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'email._header_value_parser.InvalidMessageID'
    assert len(var_2) == 0
    assert module_3.hexdigits == '0123456789abcdefABCDEF'
    assert module_3.WSP == {' ', '\t'}
    assert module_3.CFWS_LEADER == {' ', '\t', '('}
    assert module_3.SPECIALS == {'"', '@', ':', '<', ';', '\\', '[', ']', '(', ',', ')', '>', '.'}
    assert module_3.ATOM_ENDS == {' ', '"', '@', ':', '\t', '<', ';', '\\', '[', ']', '(', ',', ')', '>', '.'}
    assert module_3.DOT_ATOM_ENDS == {' ', '"', '@', ':', '\t', '<', ';', '\\', '[', ']', '(', ',', ')', '>'}
    assert module_3.PHRASE_ENDS == {'@', ':', '<', ';', '\\', '[', ']', ',', ')', '>'}
    assert module_3.TSPECIALS == {'"', '@', ':', '[', '\\', '<', ';', ']', '(', '/', ',', ')', '>', '?', '='}
    assert module_3.TOKEN_ENDS == {' ', '"', '@', ':', '\t', '<', ';', '\\', '[', ']', '(', '/', ',', ')', '>', '?', '='}
    assert module_3.ASPECIALS == {'"', '@', ':', '%', '*', '<', ';', '\\', '[', ']', "'", '(', '/', ',', ')', '>', '?', '='}
    assert module_3.ATTRIBUTE_ENDS == {'*', '[', ';', ')', '"', '%', "'", ']', ',', '>', ' ', '@', '<', '\\', '?', ':', '\t', '(', '/', '='}
    assert module_3.EXTENDED_ATTRIBUTE_ENDS == {'*', '[', ';', ')', '"', "'", ']', ',', '>', ' ', '@', '<', '\\', '?', ':', '\t', '(', '/', '='}
    assert module_3.NLSET == {'\n', '\r'}
    assert module_3.SPECIALSNL == {'\n', '"', '@', '\r', ':', '<', ';', '\\', '[', ']', '(', ',', ')', '>', '.'}
    assert f'{type(module_3.rfc2047_matcher).__module__}.{type(module_3.rfc2047_matcher).__qualname__}' == 're.Pattern'
    assert f'{type(module_3.DOT).__module__}.{type(module_3.DOT).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_3.DOT) == 1
    assert f'{type(module_3.ListSeparator).__module__}.{type(module_3.ListSeparator).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_3.ListSeparator) == 1
    assert f'{type(module_3.RouteComponentMarker).__module__}.{type(module_3.RouteComponentMarker).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_3.RouteComponentMarker) == 1
    assert module_3.InvalidMessageID.token_type == 'invalid-message-id'
    var_3 = module_0.process(var_2, var_2, raise_on_skip=var_0)
    assert var_3 is False
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
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
    var_4 = module_0.process(var_2, var_2, raise_on_skip=var_1)
    assert var_4 is False
    var_5 = [var_0]
    module_0.process(var_5, var_1)

def test_case_17():
    var_0 = 10
    var_1 = 5
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_6.Config(**var_4)
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
    assert var_5.line_length == 10
    assert var_5.wrap_length == 5
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
    assert var_5.multi_line_output == module_2.WrapModes.GRID
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
    assert len(var_5.sources) == 2
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
    assert module_6.TYPE_CHECKING is False
    assert module_6.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_6.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_6.FIRSTPARTY == 'FIRSTPARTY'
    assert module_6.FUTURE == 'FUTURE'
    assert module_6.LOCALFOLDER == 'LOCALFOLDER'
    assert module_6.STDLIB == 'STDLIB'
    assert module_6.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_6.CYTHON_EXTENSIONS).__module__}.{type(module_6.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_6.SUPPORTED_EXTENSIONS).__module__}.{type(module_6.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_6.BLOCKED_EXTENSIONS).__module__}.{type(module_6.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.BLOCKED_EXTENSIONS) == 1
    assert module_6.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_6.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_6.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_6.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_6.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_6.DEFAULT_SKIP).__module__}.{type(module_6.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_SKIP) == 19
    assert module_6.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_6.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_6.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_6.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_6.KNOWN_PREFIX == 'known_'
    assert module_6.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_6.RUNTIME_SOURCE == 'runtime'
    assert module_6.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_6.DEFAULT_CONFIG).__module__}.{type(module_6.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_6.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_6.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_6.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.skip).__module__}.{type(module_6.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_6.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_6.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_6.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_6.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_6.DEFAULT_CONFIG.skip_gitignore is False
    assert module_6.DEFAULT_CONFIG.line_length == 79
    assert module_6.DEFAULT_CONFIG.wrap_length == 0
    assert module_6.DEFAULT_CONFIG.line_ending == ''
    assert module_6.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_6.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_6.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_6.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_6.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_6.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_6.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_6.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_6.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_6.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_6.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_6.DEFAULT_CONFIG.known_other == {}
    assert module_6.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_6.DEFAULT_CONFIG.forced_separate == ()
    assert module_6.DEFAULT_CONFIG.indent == '    '
    assert module_6.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_6.DEFAULT_CONFIG.length_sort is False
    assert module_6.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_6.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_6.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.add_imports).__module__}.{type(module_6.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_6.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.remove_imports) == 0
    assert module_6.DEFAULT_CONFIG.append_only is False
    assert module_6.DEFAULT_CONFIG.reverse_relative is False
    assert module_6.DEFAULT_CONFIG.force_single_line is False
    assert module_6.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_6.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_6.DEFAULT_CONFIG.import_headings == {}
    assert module_6.DEFAULT_CONFIG.import_footers == {}
    assert module_6.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_6.DEFAULT_CONFIG.use_parentheses is False
    assert module_6.DEFAULT_CONFIG.order_by_type is True
    assert module_6.DEFAULT_CONFIG.atomic is False
    assert module_6.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_6.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_6.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_6.DEFAULT_CONFIG.lines_between_types == 0
    assert module_6.DEFAULT_CONFIG.combine_as_imports is False
    assert module_6.DEFAULT_CONFIG.combine_star is False
    assert module_6.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_6.DEFAULT_CONFIG.from_first is False
    assert module_6.DEFAULT_CONFIG.verbose is False
    assert module_6.DEFAULT_CONFIG.quiet is False
    assert module_6.DEFAULT_CONFIG.force_adds is False
    assert module_6.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_6.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_6.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_6.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_6.DEFAULT_CONFIG.lexicographical is False
    assert module_6.DEFAULT_CONFIG.group_by_package is False
    assert module_6.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_6.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_6.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_6.DEFAULT_CONFIG.no_inline_sort is False
    assert module_6.DEFAULT_CONFIG.ignore_comments is False
    assert module_6.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_6.DEFAULT_CONFIG.sources).__module__}.{type(module_6.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_6.DEFAULT_CONFIG.sources) == 1
    assert module_6.DEFAULT_CONFIG.virtual_env == ''
    assert module_6.DEFAULT_CONFIG.conda_env == ''
    assert module_6.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_6.DEFAULT_CONFIG.directory == '/workspace'
    assert module_6.DEFAULT_CONFIG.profile == ''
    assert module_6.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_6.DEFAULT_CONFIG.src_paths).__module__}.{type(module_6.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_6.DEFAULT_CONFIG.src_paths) == 2
    assert module_6.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_6.DEFAULT_CONFIG.float_to_top is False
    assert module_6.DEFAULT_CONFIG.filter_files is False
    assert module_6.DEFAULT_CONFIG.formatter == ''
    assert module_6.DEFAULT_CONFIG.formatting_function is None
    assert module_6.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_6.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_6.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_6.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_6.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_6.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_6.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_6.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_6.DEFAULT_CONFIG.constants).__module__}.{type(module_6.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.classes).__module__}.{type(module_6.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.variables).__module__}.{type(module_6.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.variables) == 0
    assert module_6.DEFAULT_CONFIG.dedup_headings is False
    assert module_6.DEFAULT_CONFIG.only_sections is False
    assert module_6.DEFAULT_CONFIG.only_modified is False
    assert module_6.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_6.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_6.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_6.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_6.DEFAULT_CONFIG.follow_links is True
    assert module_6.DEFAULT_CONFIG.indented_import_headings is True
    assert module_6.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_6.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_6.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_6.DEFAULT_CONFIG.reverse_sort is False
    assert module_6.DEFAULT_CONFIG.star_first is False
    assert module_6.DEFAULT_CONFIG.git_ls_files == {}
    assert module_6.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_6.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_6.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_6.DEFAULT_CONFIG.sort_reexports is False
    assert module_6.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_6.Config.known_patterns).__module__}.{type(module_6.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.section_comments).__module__}.{type(module_6.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.section_comments_end).__module__}.{type(module_6.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.skips).__module__}.{type(module_6.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.skip_globs).__module__}.{type(module_6.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.sorting_function).__module__}.{type(module_6.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_6 = '                    '
    var_7 = module_0._indented_config(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.settings.Config'
    assert var_7.py_version == 'py3'
    assert f'{type(var_7.force_to_top).__module__}.{type(var_7.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.force_to_top) == 0
    assert f'{type(var_7.skip).__module__}.{type(var_7.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.skip) == 19
    assert f'{type(var_7.extend_skip).__module__}.{type(var_7.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.extend_skip) == 0
    assert f'{type(var_7.skip_glob).__module__}.{type(var_7.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.skip_glob) == 0
    assert f'{type(var_7.extend_skip_glob).__module__}.{type(var_7.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.extend_skip_glob) == 0
    assert var_7.skip_gitignore is False
    assert var_7.line_length == 0
    assert var_7.wrap_length == 0
    assert var_7.line_ending == ''
    assert var_7.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_7.no_sections is False
    assert f'{type(var_7.known_future_library).__module__}.{type(var_7.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_future_library) == 1
    assert f'{type(var_7.known_third_party).__module__}.{type(var_7.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_third_party) == 0
    assert f'{type(var_7.known_first_party).__module__}.{type(var_7.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_first_party) == 0
    assert f'{type(var_7.known_local_folder).__module__}.{type(var_7.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_local_folder) == 0
    assert f'{type(var_7.known_standard_library).__module__}.{type(var_7.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_standard_library) == 234
    assert f'{type(var_7.extra_standard_library).__module__}.{type(var_7.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.extra_standard_library) == 0
    assert var_7.known_other == {}
    assert var_7.multi_line_output == module_2.WrapModes.GRID
    assert var_7.forced_separate == ()
    assert var_7.indent == '    '
    assert var_7.comment_prefix == '  #'
    assert var_7.length_sort is False
    assert var_7.length_sort_straight is False
    assert f'{type(var_7.length_sort_sections).__module__}.{type(var_7.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.length_sort_sections) == 0
    assert f'{type(var_7.add_imports).__module__}.{type(var_7.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.add_imports) == 0
    assert f'{type(var_7.remove_imports).__module__}.{type(var_7.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.remove_imports) == 0
    assert var_7.append_only is False
    assert var_7.reverse_relative is False
    assert var_7.force_single_line is False
    assert var_7.single_line_exclusions == ()
    assert var_7.default_section == 'THIRDPARTY'
    assert var_7.import_headings == {}
    assert var_7.import_footers == {}
    assert var_7.balanced_wrapping is False
    assert var_7.use_parentheses is False
    assert var_7.order_by_type is True
    assert var_7.atomic is False
    assert var_7.lines_before_imports == -1
    assert var_7.lines_after_imports == 1
    assert var_7.lines_between_sections == 1
    assert var_7.lines_between_types == 0
    assert var_7.combine_as_imports is False
    assert var_7.combine_star is False
    assert var_7.include_trailing_comma is False
    assert var_7.from_first is False
    assert var_7.verbose is False
    assert var_7.quiet is False
    assert var_7.force_adds is False
    assert var_7.force_alphabetical_sort_within_sections is False
    assert var_7.force_alphabetical_sort is False
    assert var_7.force_grid_wrap == 0
    assert var_7.force_sort_within_sections is False
    assert var_7.lexicographical is False
    assert var_7.group_by_package is False
    assert var_7.ignore_whitespace is False
    assert f'{type(var_7.no_lines_before).__module__}.{type(var_7.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.no_lines_before) == 0
    assert var_7.no_inline_sort is False
    assert var_7.ignore_comments is False
    assert var_7.case_sensitive is False
    assert f'{type(var_7.sources).__module__}.{type(var_7.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_7.sources) == 2
    assert var_7.virtual_env == ''
    assert var_7.conda_env == ''
    assert var_7.ensure_newline_before_comments is False
    assert var_7.directory == '/workspace'
    assert var_7.profile == ''
    assert var_7.honor_noqa is False
    assert f'{type(var_7.src_paths).__module__}.{type(var_7.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_7.src_paths) == 2
    assert var_7.remove_redundant_aliases is False
    assert var_7.float_to_top is False
    assert var_7.filter_files is False
    assert var_7.formatter == ''
    assert var_7.formatting_function is None
    assert var_7.color_output is False
    assert f'{type(var_7.treat_comments_as_code).__module__}.{type(var_7.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.treat_comments_as_code) == 0
    assert var_7.treat_all_comments_as_code is False
    assert f'{type(var_7.supported_extensions).__module__}.{type(var_7.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.supported_extensions) == 4
    assert f'{type(var_7.blocked_extensions).__module__}.{type(var_7.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.blocked_extensions) == 1
    assert f'{type(var_7.constants).__module__}.{type(var_7.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.constants) == 0
    assert f'{type(var_7.classes).__module__}.{type(var_7.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.classes) == 0
    assert f'{type(var_7.variables).__module__}.{type(var_7.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.variables) == 0
    assert var_7.dedup_headings is False
    assert var_7.only_sections is False
    assert var_7.only_modified is False
    assert var_7.combine_straight_imports is False
    assert var_7.auto_identify_namespace_packages is True
    assert f'{type(var_7.namespace_packages).__module__}.{type(var_7.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.namespace_packages) == 0
    assert var_7.follow_links is True
    assert var_7.indented_import_headings is True
    assert var_7.honor_case_in_force_sorted_sections is False
    assert var_7.sort_relative_in_force_sorted_sections is False
    assert var_7.overwrite_in_place is False
    assert var_7.reverse_sort is False
    assert var_7.star_first is False
    assert var_7.git_ls_files == {}
    assert var_7.format_error == '{error}: {message}'
    assert var_7.format_success == '{success}: {message}'
    assert var_7.sort_order == 'natural'
    assert var_7.sort_reexports is False
    assert var_7.split_on_trailing_comma is False
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
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
    var_8 = var_7.line_length
    assert var_8 == 0
    var_9 = var_7.wrap_length
    assert var_9 == 0

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 88
    var_1 = 79
    var_2 = False
    var_3 = 'line_length'
    var_4 = 'wrap_length'
    var_5 = 'indented_import_headings'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_6.Config(**var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.settings.Config'
    assert var_7.py_version == 'py3'
    assert f'{type(var_7.force_to_top).__module__}.{type(var_7.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.force_to_top) == 0
    assert f'{type(var_7.skip).__module__}.{type(var_7.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.skip) == 19
    assert f'{type(var_7.extend_skip).__module__}.{type(var_7.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.extend_skip) == 0
    assert f'{type(var_7.skip_glob).__module__}.{type(var_7.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.skip_glob) == 0
    assert f'{type(var_7.extend_skip_glob).__module__}.{type(var_7.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.extend_skip_glob) == 0
    assert var_7.skip_gitignore is False
    assert var_7.line_length == 88
    assert var_7.wrap_length == 79
    assert var_7.line_ending == ''
    assert var_7.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_7.no_sections is False
    assert f'{type(var_7.known_future_library).__module__}.{type(var_7.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_future_library) == 1
    assert f'{type(var_7.known_third_party).__module__}.{type(var_7.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_third_party) == 0
    assert f'{type(var_7.known_first_party).__module__}.{type(var_7.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_first_party) == 0
    assert f'{type(var_7.known_local_folder).__module__}.{type(var_7.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_local_folder) == 0
    assert f'{type(var_7.known_standard_library).__module__}.{type(var_7.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_standard_library) == 234
    assert f'{type(var_7.extra_standard_library).__module__}.{type(var_7.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.extra_standard_library) == 0
    assert var_7.known_other == {}
    assert var_7.multi_line_output == module_2.WrapModes.GRID
    assert var_7.forced_separate == ()
    assert var_7.indent == '    '
    assert var_7.comment_prefix == '  #'
    assert var_7.length_sort is False
    assert var_7.length_sort_straight is False
    assert f'{type(var_7.length_sort_sections).__module__}.{type(var_7.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.length_sort_sections) == 0
    assert f'{type(var_7.add_imports).__module__}.{type(var_7.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.add_imports) == 0
    assert f'{type(var_7.remove_imports).__module__}.{type(var_7.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.remove_imports) == 0
    assert var_7.append_only is False
    assert var_7.reverse_relative is False
    assert var_7.force_single_line is False
    assert var_7.single_line_exclusions == ()
    assert var_7.default_section == 'THIRDPARTY'
    assert var_7.import_headings == {}
    assert var_7.import_footers == {}
    assert var_7.balanced_wrapping is False
    assert var_7.use_parentheses is False
    assert var_7.order_by_type is True
    assert var_7.atomic is False
    assert var_7.lines_before_imports == -1
    assert var_7.lines_after_imports == -1
    assert var_7.lines_between_sections == 1
    assert var_7.lines_between_types == 0
    assert var_7.combine_as_imports is False
    assert var_7.combine_star is False
    assert var_7.include_trailing_comma is False
    assert var_7.from_first is False
    assert var_7.verbose is False
    assert var_7.quiet is False
    assert var_7.force_adds is False
    assert var_7.force_alphabetical_sort_within_sections is False
    assert var_7.force_alphabetical_sort is False
    assert var_7.force_grid_wrap == 0
    assert var_7.force_sort_within_sections is False
    assert var_7.lexicographical is False
    assert var_7.group_by_package is False
    assert var_7.ignore_whitespace is False
    assert f'{type(var_7.no_lines_before).__module__}.{type(var_7.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.no_lines_before) == 0
    assert var_7.no_inline_sort is False
    assert var_7.ignore_comments is False
    assert var_7.case_sensitive is False
    assert f'{type(var_7.sources).__module__}.{type(var_7.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_7.sources) == 2
    assert var_7.virtual_env == ''
    assert var_7.conda_env == ''
    assert var_7.ensure_newline_before_comments is False
    assert var_7.directory == '/workspace'
    assert var_7.profile == ''
    assert var_7.honor_noqa is False
    assert f'{type(var_7.src_paths).__module__}.{type(var_7.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_7.src_paths) == 2
    assert var_7.remove_redundant_aliases is False
    assert var_7.float_to_top is False
    assert var_7.filter_files is False
    assert var_7.formatter == ''
    assert var_7.formatting_function is None
    assert var_7.color_output is False
    assert f'{type(var_7.treat_comments_as_code).__module__}.{type(var_7.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.treat_comments_as_code) == 0
    assert var_7.treat_all_comments_as_code is False
    assert f'{type(var_7.supported_extensions).__module__}.{type(var_7.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.supported_extensions) == 4
    assert f'{type(var_7.blocked_extensions).__module__}.{type(var_7.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.blocked_extensions) == 1
    assert f'{type(var_7.constants).__module__}.{type(var_7.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.constants) == 0
    assert f'{type(var_7.classes).__module__}.{type(var_7.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.classes) == 0
    assert f'{type(var_7.variables).__module__}.{type(var_7.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.variables) == 0
    assert var_7.dedup_headings is False
    assert var_7.only_sections is False
    assert var_7.only_modified is False
    assert var_7.combine_straight_imports is False
    assert var_7.auto_identify_namespace_packages is True
    assert f'{type(var_7.namespace_packages).__module__}.{type(var_7.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.namespace_packages) == 0
    assert var_7.follow_links is True
    assert var_7.indented_import_headings is False
    assert var_7.honor_case_in_force_sorted_sections is False
    assert var_7.sort_relative_in_force_sorted_sections is False
    assert var_7.overwrite_in_place is False
    assert var_7.reverse_sort is False
    assert var_7.star_first is False
    assert var_7.git_ls_files == {}
    assert var_7.format_error == '{error}: {message}'
    assert var_7.format_success == '{success}: {message}'
    assert var_7.sort_order == 'natural'
    assert var_7.sort_reexports is False
    assert var_7.split_on_trailing_comma is False
    assert module_6.TYPE_CHECKING is False
    assert module_6.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_6.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_6.FIRSTPARTY == 'FIRSTPARTY'
    assert module_6.FUTURE == 'FUTURE'
    assert module_6.LOCALFOLDER == 'LOCALFOLDER'
    assert module_6.STDLIB == 'STDLIB'
    assert module_6.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_6.CYTHON_EXTENSIONS).__module__}.{type(module_6.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_6.SUPPORTED_EXTENSIONS).__module__}.{type(module_6.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_6.BLOCKED_EXTENSIONS).__module__}.{type(module_6.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.BLOCKED_EXTENSIONS) == 1
    assert module_6.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_6.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_6.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_6.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_6.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_6.DEFAULT_SKIP).__module__}.{type(module_6.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_SKIP) == 19
    assert module_6.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_6.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_6.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_6.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_6.KNOWN_PREFIX == 'known_'
    assert module_6.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_6.RUNTIME_SOURCE == 'runtime'
    assert module_6.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_6.DEFAULT_CONFIG).__module__}.{type(module_6.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_6.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_6.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_6.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.skip).__module__}.{type(module_6.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_6.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_6.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_6.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_6.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_6.DEFAULT_CONFIG.skip_gitignore is False
    assert module_6.DEFAULT_CONFIG.line_length == 79
    assert module_6.DEFAULT_CONFIG.wrap_length == 0
    assert module_6.DEFAULT_CONFIG.line_ending == ''
    assert module_6.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_6.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_6.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_6.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_6.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_6.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_6.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_6.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_6.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_6.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_6.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_6.DEFAULT_CONFIG.known_other == {}
    assert module_6.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_6.DEFAULT_CONFIG.forced_separate == ()
    assert module_6.DEFAULT_CONFIG.indent == '    '
    assert module_6.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_6.DEFAULT_CONFIG.length_sort is False
    assert module_6.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_6.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_6.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.add_imports).__module__}.{type(module_6.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_6.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.remove_imports) == 0
    assert module_6.DEFAULT_CONFIG.append_only is False
    assert module_6.DEFAULT_CONFIG.reverse_relative is False
    assert module_6.DEFAULT_CONFIG.force_single_line is False
    assert module_6.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_6.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_6.DEFAULT_CONFIG.import_headings == {}
    assert module_6.DEFAULT_CONFIG.import_footers == {}
    assert module_6.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_6.DEFAULT_CONFIG.use_parentheses is False
    assert module_6.DEFAULT_CONFIG.order_by_type is True
    assert module_6.DEFAULT_CONFIG.atomic is False
    assert module_6.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_6.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_6.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_6.DEFAULT_CONFIG.lines_between_types == 0
    assert module_6.DEFAULT_CONFIG.combine_as_imports is False
    assert module_6.DEFAULT_CONFIG.combine_star is False
    assert module_6.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_6.DEFAULT_CONFIG.from_first is False
    assert module_6.DEFAULT_CONFIG.verbose is False
    assert module_6.DEFAULT_CONFIG.quiet is False
    assert module_6.DEFAULT_CONFIG.force_adds is False
    assert module_6.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_6.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_6.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_6.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_6.DEFAULT_CONFIG.lexicographical is False
    assert module_6.DEFAULT_CONFIG.group_by_package is False
    assert module_6.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_6.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_6.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_6.DEFAULT_CONFIG.no_inline_sort is False
    assert module_6.DEFAULT_CONFIG.ignore_comments is False
    assert module_6.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_6.DEFAULT_CONFIG.sources).__module__}.{type(module_6.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_6.DEFAULT_CONFIG.sources) == 1
    assert module_6.DEFAULT_CONFIG.virtual_env == ''
    assert module_6.DEFAULT_CONFIG.conda_env == ''
    assert module_6.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_6.DEFAULT_CONFIG.directory == '/workspace'
    assert module_6.DEFAULT_CONFIG.profile == ''
    assert module_6.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_6.DEFAULT_CONFIG.src_paths).__module__}.{type(module_6.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_6.DEFAULT_CONFIG.src_paths) == 2
    assert module_6.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_6.DEFAULT_CONFIG.float_to_top is False
    assert module_6.DEFAULT_CONFIG.filter_files is False
    assert module_6.DEFAULT_CONFIG.formatter == ''
    assert module_6.DEFAULT_CONFIG.formatting_function is None
    assert module_6.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_6.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_6.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_6.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_6.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_6.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_6.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_6.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_6.DEFAULT_CONFIG.constants).__module__}.{type(module_6.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.classes).__module__}.{type(module_6.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.variables).__module__}.{type(module_6.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.variables) == 0
    assert module_6.DEFAULT_CONFIG.dedup_headings is False
    assert module_6.DEFAULT_CONFIG.only_sections is False
    assert module_6.DEFAULT_CONFIG.only_modified is False
    assert module_6.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_6.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_6.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_6.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_6.DEFAULT_CONFIG.follow_links is True
    assert module_6.DEFAULT_CONFIG.indented_import_headings is True
    assert module_6.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_6.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_6.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_6.DEFAULT_CONFIG.reverse_sort is False
    assert module_6.DEFAULT_CONFIG.star_first is False
    assert module_6.DEFAULT_CONFIG.git_ls_files == {}
    assert module_6.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_6.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_6.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_6.DEFAULT_CONFIG.sort_reexports is False
    assert module_6.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_6.Config.known_patterns).__module__}.{type(module_6.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.section_comments).__module__}.{type(module_6.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.section_comments_end).__module__}.{type(module_6.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.skips).__module__}.{type(module_6.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.skip_globs).__module__}.{type(module_6.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_6.Config.sorting_function).__module__}.{type(module_6.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_8 = '    '
    var_9 = module_0._indented_config(var_7, var_8)
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
    assert var_9.line_length == 84
    assert var_9.wrap_length == 75
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
    assert var_9.multi_line_output == module_2.WrapModes.GRID
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
    assert var_9.lines_after_imports == 1
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
    assert len(var_9.sources) == 2
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
    assert var_9.indented_import_headings is False
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
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
    var_10 = var_9.line_length
    assert var_10 == 84
    var_11 = var_9.__repr__()
    assert var_11 == "Config(py_version='py3', force_to_top=frozenset(), skip=frozenset({'.pytype', 'dist', '.eggs', '.venv', '.direnv', 'build', '.svn', '.tox', '__pypackages__', '.hg', '.mypy_cache', 'buck-out', '.nox', 'node_modules', '.pants.d', '_build', '.git', 'venv', '.bzr'}), extend_skip=frozenset(), skip_glob=frozenset(), extend_skip_glob=frozenset(), skip_gitignore=False, line_length=84, wrap_length=75, line_ending='', sections=('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'), no_sections=False, known_future_library=frozenset({'__future__'}), known_third_party=frozenset(), known_first_party=frozenset(), known_local_folder=frozenset(), known_standard_library=frozenset({'xdrlib', 'filecmp', 'time', 'subprocess', 'compileall', 'logging', 'asyncio', 'shlex', 'itertools', 'base64', 'socket', 'zipfile', 'cgi', 'pty', 'curses', 'plistlib', 'dbm', 'argparse', 'nntplib', 'enum', 'stringprep', 'hashlib', 'warnings', 'profile', 'runpy', 'tracemalloc', 'urllib', 'os', 'copy', 'netrc', 'chunk', 'abc', 'sndhdr', 'dataclasses', 'fractions', 'lzma', 'sre_compile', 'codeop', 'optparse', 'bisect', 'crypt', 'graphlib', 'socketserver', 'pwd', 'turtledemo', 'asyncore', 'linecache', 'timeit', 'reprlib', 'tkinter', 'site', 'bdb', 'pprint', 'pyclbr', 'functools', 'quopri', 'tabnanny', 'sunau', 'xml', 'doctest', 'datetime', 'smtpd', 'sysconfig', 'ipaddress', 'shelve', 'http', 'hmac', 'inspect', 'keyword', 'zipapp', 'spwd', 'uu', 'dis', 'cProfile', 'rlcompleter', 'concurrent', 'email', 'readline', 'turtle', 'ntpath', 'gc', 'wave', 'uuid', 'cmd', 'tarfile', 'genericpath', 'symtable', 'copyreg', 'pkgutil', 'shutil', 'mailbox', 'ctypes', '_thread', 'decimal', 'macpath', 'xmlrpc', 'resource', '_dummy_thread', 'numbers', 'struct', 'select', 'imaplib', 'stat', 'ftplib', 'this', 'weakref', 'poplib', 'pstats', 'lib2to3', 'telnetlib', 'ast', 'queue', 'smtplib', 'xxlimited', 'threading', 'unittest', 'string', 'grp', 'textwrap', 'posixpath', 'fcntl', 'gzip', 'nis', 'multiprocessing', 'signal', 'fileinput', 'test', 'token', 'ensurepip', '_ast', 'modulefinder', 'marshal', 'tomllib', 'platform', 'posix', 'sys', 'configparser', 'xx', 'traceback', 'sre_constants', 'xxsubtype', 'pipes', 'glob', 'math', 'antigravity', 'locale', 'binhex', 'pydoc', 'msilib', 'collections', 'webbrowser', 'ossaudiodev', 'html', 'contextvars', 'csv', 'unicodedata', 'sqlite3', 'tty', 're', 'parser', 'pickle', 'zlib', 'nturl2path', 'pyexpat', 'zoneinfo', 'io', 'pdb', 'operator', 'pickletools', 'asynchat', 'json', 'imp', 'pydoc_data', 'msvcrt', 'atexit', 'secrets', 'annotationlib', 'gettext', 'sre_parse', 'errno', 'builtins', 'imghdr', 'sre', 'xxlimited_35', 'random', 'aifc', 'winsound', 'importlib', 'cgitb', 'fpectl', 'idlelib', 'py_compile', 'mmap', 'calendar', 'ssl', 'pathlib', 'encodings', 'getopt', 'sched', 'distutils', 'selectors', 'bz2', 'venv', 'dummy_threading', 'types', 'faulthandler', 'opcode', 'formatter', 'getpass', 'binascii', 'symbol', 'tokenize', 'statistics', 'winreg', 'typing', 'array', 'contextlib', 'syslog', 'colorsys', 'wsgiref', 'fnmatch', 'heapq', 'mimetypes', 'cmath', 'tempfile', 'mailcap', 'termios', 'zipimport', 'nt', 'code', 'trace', 'audioop', 'codecs', 'difflib'}), extra_standard_library=frozenset(), known_other={}, multi_line_output=<WrapModes.GRID: 0>, forced_separate=(), indent='    ', comment_prefix='  #', length_sort=False, length_sort_straight=False, length_sort_sections=frozenset(), add_imports=frozenset(), remove_imports=frozenset(), append_only=False, reverse_relative=False, force_single_line=False, single_line_exclusions=(), default_section='THIRDPARTY', import_headings={}, import_footers={}, balanced_wrapping=False, use_parentheses=False, order_by_type=True, atomic=False, lines_before_imports=-1, lines_after_imports=1, lines_between_sections=1, lines_between_types=0, combine_as_imports=False, combine_star=False, include_trailing_comma=False, from_first=False, verbose=False, quiet=False, force_adds=False, force_alphabetical_sort_within_sections=False, force_alphabetical_sort=False, force_grid_wrap=0, force_sort_within_sections=False, lexicographical=False, group_by_package=False, ignore_whitespace=False, no_lines_before=frozenset(), no_inline_sort=False, ignore_comments=False, case_sensitive=False, sources=({'py_version': 'py3', 'force_to_top': frozenset(), 'skip': frozenset({'.pytype', 'dist', '.eggs', '.venv', '.direnv', 'build', '.svn', '.tox', '__pypackages__', '.hg', '.mypy_cache', 'buck-out', '.nox', 'node_modules', '.pants.d', '_build', '.git', 'venv', '.bzr'}), 'extend_skip': frozenset(), 'skip_glob': frozenset(), 'extend_skip_glob': frozenset(), 'skip_gitignore': False, 'line_length': 79, 'wrap_length': 0, 'line_ending': '', 'sections': ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'), 'no_sections': False, 'known_future_library': frozenset({'__future__'}), 'known_third_party': frozenset(), 'known_first_party': frozenset(), 'known_local_folder': frozenset(), 'known_standard_library': frozenset({'xdrlib', 'filecmp', 'time', 'subprocess', 'compileall', 'logging', 'asyncio', 'shlex', 'itertools', 'base64', 'socket', 'zipfile', 'cgi', 'pty', 'curses', 'plistlib', 'dbm', 'argparse', 'nntplib', 'enum', 'stringprep', 'hashlib', 'warnings', 'profile', 'runpy', 'tracemalloc', 'urllib', 'os', 'copy', 'netrc', 'chunk', 'abc', 'sndhdr', 'dataclasses', 'fractions', 'lzma', 'sre_compile', 'codeop', 'optparse', 'bisect', 'crypt', 'graphlib', 'socketserver', 'pwd', 'turtledemo', 'asyncore', 'linecache', 'timeit', 'reprlib', 'tkinter', 'site', 'bdb', 'pprint', 'pyclbr', 'functools', 'quopri', 'tabnanny', 'sunau', 'xml', 'doctest', 'datetime', 'smtpd', 'sysconfig', 'ipaddress', 'shelve', 'http', 'hmac', 'inspect', 'keyword', 'zipapp', 'spwd', 'uu', 'dis', 'cProfile', 'rlcompleter', 'concurrent', 'email', 'readline', 'turtle', 'ntpath', 'gc', 'wave', 'uuid', 'cmd', 'tarfile', 'genericpath', 'symtable', 'copyreg', 'pkgutil', 'shutil', 'mailbox', 'ctypes', '_thread', 'decimal', 'macpath', 'xmlrpc', 'resource', '_dummy_thread', 'numbers', 'struct', 'select', 'imaplib', 'stat', 'ftplib', 'this', 'weakref', 'poplib', 'pstats', 'lib2to3', 'telnetlib', 'ast', 'queue', 'smtplib', 'xxlimited', 'threading', 'unittest', 'string', 'grp', 'textwrap', 'posixpath', 'fcntl', 'gzip', 'nis', 'multiprocessing', 'signal', 'fileinput', 'test', 'token', 'ensurepip', '_ast', 'modulefinder', 'marshal', 'tomllib', 'platform', 'posix', 'sys', 'configparser', 'xx', 'traceback', 'sre_constants', 'xxsubtype', 'pipes', 'glob', 'math', 'antigravity', 'locale', 'binhex', 'pydoc', 'msilib', 'collections', 'webbrowser', 'ossaudiodev', 'html', 'contextvars', 'csv', 'unicodedata', 'sqlite3', 'tty', 're', 'parser', 'pickle', 'zlib', 'nturl2path', 'pyexpat', 'zoneinfo', 'io', 'pdb', 'operator', 'pickletools', 'asynchat', 'json', 'imp', 'pydoc_data', 'msvcrt', 'atexit', 'secrets', 'annotationlib', 'gettext', 'sre_parse', 'errno', 'builtins', 'imghdr', 'sre', 'xxlimited_35', 'random', 'aifc', 'winsound', 'importlib', 'cgitb', 'fpectl', 'idlelib', 'py_compile', 'mmap', 'calendar', 'ssl', 'pathlib', 'encodings', 'getopt', 'sched', 'distutils', 'selectors', 'bz2', 'venv', 'dummy_threading', 'types', 'faulthandler', 'opcode', 'formatter', 'getpass', 'binascii', 'symbol', 'tokenize', 'statistics', 'winreg', 'typing', 'array', 'contextlib', 'syslog', 'colorsys', 'wsgiref', 'fnmatch', 'heapq', 'mimetypes', 'cmath', 'tempfile', 'mailcap', 'termios', 'zipimport', 'nt', 'code', 'trace', 'audioop', 'codecs', 'difflib'}), 'extra_standard_library': frozenset(), 'known_other': {}, 'multi_line_output': <WrapModes.GRID: 0>, 'forced_separate': (), 'indent': '    ', 'comment_prefix': '  #', 'length_sort': False, 'length_sort_straight': False, 'length_sort_sections': frozenset(), 'add_imports': frozenset(), 'remove_imports': frozenset(), 'append_only': False, 'reverse_relative': False, 'force_single_line': False, 'single_line_exclusions': (), 'default_section': 'THIRDPARTY', 'import_headings': {}, 'import_footers': {}, 'balanced_wrapping': False, 'use_parentheses': False, 'order_by_type': True, 'atomic': False, 'lines_before_imports': -1, 'lines_after_imports': -1, 'lines_between_sections': 1, 'lines_between_types': 0, 'combine_as_imports': False, 'combine_star': False, 'include_trailing_comma': False, 'from_first': False, 'verbose': False, 'quiet': False, 'force_adds': False, 'force_alphabetical_sort_within_sections': False, 'force_alphabetical_sort': False, 'force_grid_wrap': 0, 'force_sort_within_sections': False, 'lexicographical': False, 'group_by_package': False, 'ignore_whitespace': False, 'no_lines_before': frozenset(), 'no_inline_sort': False, 'ignore_comments': False, 'case_sensitive': False, 'sources': (), 'virtual_env': '', 'conda_env': '', 'ensure_newline_before_comments': False, 'directory': '', 'profile': '', 'honor_noqa': False, 'src_paths': (), 'remove_redundant_aliases': False, 'float_to_top': False, 'filter_files': False, 'formatter': '', 'formatting_function': None, 'color_output': False, 'treat_comments_as_code': frozenset(), 'treat_all_comments_as_code': False, 'supported_extensions': frozenset({'pyx', 'pyi', 'py', 'pxd'}), 'blocked_extensions': frozenset({'pex'}), 'constants': frozenset(), 'classes': frozenset(), 'variables': frozenset(), 'dedup_headings': False, 'only_sections': False, 'only_modified': False, 'combine_straight_imports': False, 'auto_identify_namespace_packages': True, 'namespace_packages': frozenset(), 'follow_links': True, 'indented_import_headings': True, 'honor_case_in_force_sorted_sections': False, 'sort_relative_in_force_sorted_sections': False, 'overwrite_in_place': False, 'reverse_sort': False, 'star_first': False, 'git_ls_files': {}, 'format_error': '{error}: {message}', 'format_success': '{success}: {message}', 'sort_order': 'natural', 'sort_reexports': False, 'split_on_trailing_comma': False, 'source': 'defaults'}, {'line_length': 88, 'wrap_length': 79, 'indented_import_headings': False, 'source': 'runtime'}), virtual_env='', conda_env='', ensure_newline_before_comments=False, directory='/workspace', profile='', honor_noqa=False, src_paths=(PosixPath('/workspace/src'), PosixPath('/workspace')), remove_redundant_aliases=False, float_to_top=False, filter_files=False, formatter='', formatting_function=None, color_output=False, treat_comments_as_code=frozenset(), treat_all_comments_as_code=False, supported_extensions=frozenset({'pyx', 'pyi', 'py', 'pxd'}), blocked_extensions=frozenset({'pex'}), constants=frozenset(), classes=frozenset(), variables=frozenset(), dedup_headings=False, only_sections=False, only_modified=False, combine_straight_imports=False, auto_identify_namespace_packages=True, namespace_packages=frozenset(), follow_links=True, indented_import_headings=False, honor_case_in_force_sorted_sections=False, sort_relative_in_force_sorted_sections=False, overwrite_in_place=False, reverse_sort=False, star_first=False, git_ls_files={}, format_error='{error}: {message}', format_success='{success}: {message}', sort_order='natural', sort_reexports=False, split_on_trailing_comma=False)"
    module_0.process(var_10, var_10)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = "|?\t\x0bpK(#b-`z'd"
    var_1 = [var_0, var_0]
    module_0.process(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'from <y import argv\nfrom os import path\n'
    var_1 = module_4.splitext(var_0)
    assert module_4.curdir == '.'
    assert module_4.pardir == '..'
    assert module_4.extsep == '.'
    assert module_4.sep == '/'
    assert module_4.pathsep == ':'
    assert module_4.defpath == '/bin:/usr/bin'
    assert module_4.altsep is None
    assert module_4.devnull == '/dev/null'
    assert f'{type(module_4.ALLOW_MISSING).__module__}.{type(module_4.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    assert module_4.supports_unicode_filenames is False
    module_0.process(var_1, var_1)