# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.providers.development as module_0
import mimesis.providers.base as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0.Development()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.LICENSES == ['Apache License, 2.0 (Apache-2.0)', 'The BSD 3-Clause License', 'The BSD 2-Clause License', 'GNU General Public License (GPL)', 'General Public License (LGPL)', 'MIT License (MIT)', 'Mozilla Public License 2.0 (MPL-2.0)', 'Common Development and Distribution License (CDDL-1.0)', 'Eclipse Public License (EPL-1.0)']
    assert module_0.OS == ['Arch', 'CentOS', 'Debian', 'Fedora', 'FreeBSD', 'Gentoo', 'Kali', 'Lubuntu', 'Manjaro', 'Mint', 'OS X', 'macOS', 'OpenBSD', 'Slackware', 'Ubuntu', 'Windows 10', 'Windows 7', 'Windows 8', 'Windows 8.1', 'Windows 11', 'elementaryOS', 'macOS', 'openSUSE']
    assert module_0.PROGRAMMING_LANGS == ['ASP', 'Assembly', 'AutoIt', 'Awk', 'Bash', 'C', 'C Shell', 'C#', 'C++', 'Caml', 'Ceylon', 'Clojure', 'CoffeeScript', 'Common Lisp', 'D', 'Dart', 'Delphi', 'Dylan', 'ECMAScript', 'Elixir', 'Emacs Lisp', 'Erlang', 'F#', 'Falcon', 'Fortran', 'GNU Octave', 'Go', 'Groovy', 'Haskell', 'haXe', 'Io', 'J#', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'Lisp', 'Lua', 'Mathematica', 'Objective-C', 'OCaml', 'Perl', 'PHP', 'PL-I', 'PL-SQL', 'PowerShell', 'Prolog', 'Python', 'R', 'Racket', 'Ruby', 'Rust', 'Scala', 'Scheme', 'Smalltalk', 'Tcl', 'Tex', 'Transact-SQL', 'TypeScript', 'Z shell']
    assert module_0.STAGES == ('Pre-alpha', 'Alpha', 'Beta', 'RC', 'Stable')
    assert module_0.SYSTEM_QUALITY_ATTRIBUTES == ('accessibility', 'accountability', 'accuracy', 'adaptability', 'administrability', 'affordability', 'agility', 'auditability', 'autonomy', 'availability', 'compatibility', 'composability', 'confidentiality', 'configurability', 'correctness', 'credibility', 'customizability', 'debuggability', 'degradability', 'demonstrability', 'dependability', 'deployability', 'determinability', 'discoverability', 'distributability', 'durability', 'effectiveness', 'efficiency', 'evolvability', 'extensibility', 'failure transparency', 'fault-tolerance', 'fidelity', 'flexibility', 'inspectability', 'installability', 'integrity', 'interchangeability', 'interoperability', 'learnability', 'localizability', 'maintainability', 'manageability', 'mobility', 'modifiability', 'modularity', 'observability', 'operability', 'orthogonality', 'portability', 'precision', 'predictability', 'process capabilities', 'producibility', 'provability', 'recoverability', 'redundancy', 'relevance', 'reliability', 'repeatability', 'reproducibility', 'resilience', 'responsiveness', 'reusability', 'robustness', 'safety', 'scalability', 'seamlessness', 'securability', 'self-sustainability', 'serviceability', 'simplicity', 'stability', 'standards compliance', 'survivability', 'sustainability', 'tailorability', 'testability', 'timeliness', 'traceability', 'transparency', 'ubiquity', 'understandability', 'upgradability', 'usability', 'vulnerability')
    var_1.validate_enum(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_1.BaseProvider()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.base.BaseProvider'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.DATADIR).__module__}.{type(module_1.DATADIR).__qualname__}' == 'pathlib.PosixPath'
    assert module_1.LOCALE_SEP == '-'
    assert f'{type(module_1.MissingSeed).__module__}.{type(module_1.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_1.Seed).__module__}.{type(module_1.Seed).__qualname__}' == 'types.UnionType'
    var_2 = module_0.Development()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_2.random).__module__}.{type(var_2.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_2.seed).__module__}.{type(var_2.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.LICENSES == ['Apache License, 2.0 (Apache-2.0)', 'The BSD 3-Clause License', 'The BSD 2-Clause License', 'GNU General Public License (GPL)', 'General Public License (LGPL)', 'MIT License (MIT)', 'Mozilla Public License 2.0 (MPL-2.0)', 'Common Development and Distribution License (CDDL-1.0)', 'Eclipse Public License (EPL-1.0)']
    assert module_0.OS == ['Arch', 'CentOS', 'Debian', 'Fedora', 'FreeBSD', 'Gentoo', 'Kali', 'Lubuntu', 'Manjaro', 'Mint', 'OS X', 'macOS', 'OpenBSD', 'Slackware', 'Ubuntu', 'Windows 10', 'Windows 7', 'Windows 8', 'Windows 8.1', 'Windows 11', 'elementaryOS', 'macOS', 'openSUSE']
    assert module_0.PROGRAMMING_LANGS == ['ASP', 'Assembly', 'AutoIt', 'Awk', 'Bash', 'C', 'C Shell', 'C#', 'C++', 'Caml', 'Ceylon', 'Clojure', 'CoffeeScript', 'Common Lisp', 'D', 'Dart', 'Delphi', 'Dylan', 'ECMAScript', 'Elixir', 'Emacs Lisp', 'Erlang', 'F#', 'Falcon', 'Fortran', 'GNU Octave', 'Go', 'Groovy', 'Haskell', 'haXe', 'Io', 'J#', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'Lisp', 'Lua', 'Mathematica', 'Objective-C', 'OCaml', 'Perl', 'PHP', 'PL-I', 'PL-SQL', 'PowerShell', 'Prolog', 'Python', 'R', 'Racket', 'Ruby', 'Rust', 'Scala', 'Scheme', 'Smalltalk', 'Tcl', 'Tex', 'Transact-SQL', 'TypeScript', 'Z shell']
    assert module_0.STAGES == ('Pre-alpha', 'Alpha', 'Beta', 'RC', 'Stable')
    assert module_0.SYSTEM_QUALITY_ATTRIBUTES == ('accessibility', 'accountability', 'accuracy', 'adaptability', 'administrability', 'affordability', 'agility', 'auditability', 'autonomy', 'availability', 'compatibility', 'composability', 'confidentiality', 'configurability', 'correctness', 'credibility', 'customizability', 'debuggability', 'degradability', 'demonstrability', 'dependability', 'deployability', 'determinability', 'discoverability', 'distributability', 'durability', 'effectiveness', 'efficiency', 'evolvability', 'extensibility', 'failure transparency', 'fault-tolerance', 'fidelity', 'flexibility', 'inspectability', 'installability', 'integrity', 'interchangeability', 'interoperability', 'learnability', 'localizability', 'maintainability', 'manageability', 'mobility', 'modifiability', 'modularity', 'observability', 'operability', 'orthogonality', 'portability', 'precision', 'predictability', 'process capabilities', 'producibility', 'provability', 'recoverability', 'redundancy', 'relevance', 'reliability', 'repeatability', 'reproducibility', 'resilience', 'responsiveness', 'reusability', 'robustness', 'safety', 'scalability', 'seamlessness', 'securability', 'self-sustainability', 'serviceability', 'simplicity', 'stability', 'standards compliance', 'survivability', 'sustainability', 'tailorability', 'testability', 'timeliness', 'traceability', 'transparency', 'ubiquity', 'understandability', 'upgradability', 'usability', 'vulnerability')
    var_3 = var_2.boolean()
    var_4 = var_2.software_license()
    var_5 = var_2.software_license()
    var_1.validate_enum(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.Development()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_1.random).__module__}.{type(var_1.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.LICENSES == ['Apache License, 2.0 (Apache-2.0)', 'The BSD 3-Clause License', 'The BSD 2-Clause License', 'GNU General Public License (GPL)', 'General Public License (LGPL)', 'MIT License (MIT)', 'Mozilla Public License 2.0 (MPL-2.0)', 'Common Development and Distribution License (CDDL-1.0)', 'Eclipse Public License (EPL-1.0)']
    assert module_0.OS == ['Arch', 'CentOS', 'Debian', 'Fedora', 'FreeBSD', 'Gentoo', 'Kali', 'Lubuntu', 'Manjaro', 'Mint', 'OS X', 'macOS', 'OpenBSD', 'Slackware', 'Ubuntu', 'Windows 10', 'Windows 7', 'Windows 8', 'Windows 8.1', 'Windows 11', 'elementaryOS', 'macOS', 'openSUSE']
    assert module_0.PROGRAMMING_LANGS == ['ASP', 'Assembly', 'AutoIt', 'Awk', 'Bash', 'C', 'C Shell', 'C#', 'C++', 'Caml', 'Ceylon', 'Clojure', 'CoffeeScript', 'Common Lisp', 'D', 'Dart', 'Delphi', 'Dylan', 'ECMAScript', 'Elixir', 'Emacs Lisp', 'Erlang', 'F#', 'Falcon', 'Fortran', 'GNU Octave', 'Go', 'Groovy', 'Haskell', 'haXe', 'Io', 'J#', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'Lisp', 'Lua', 'Mathematica', 'Objective-C', 'OCaml', 'Perl', 'PHP', 'PL-I', 'PL-SQL', 'PowerShell', 'Prolog', 'Python', 'R', 'Racket', 'Ruby', 'Rust', 'Scala', 'Scheme', 'Smalltalk', 'Tcl', 'Tex', 'Transact-SQL', 'TypeScript', 'Z shell']
    assert module_0.STAGES == ('Pre-alpha', 'Alpha', 'Beta', 'RC', 'Stable')
    assert module_0.SYSTEM_QUALITY_ATTRIBUTES == ('accessibility', 'accountability', 'accuracy', 'adaptability', 'administrability', 'affordability', 'agility', 'auditability', 'autonomy', 'availability', 'compatibility', 'composability', 'confidentiality', 'configurability', 'correctness', 'credibility', 'customizability', 'debuggability', 'degradability', 'demonstrability', 'dependability', 'deployability', 'determinability', 'discoverability', 'distributability', 'durability', 'effectiveness', 'efficiency', 'evolvability', 'extensibility', 'failure transparency', 'fault-tolerance', 'fidelity', 'flexibility', 'inspectability', 'installability', 'integrity', 'interchangeability', 'interoperability', 'learnability', 'localizability', 'maintainability', 'manageability', 'mobility', 'modifiability', 'modularity', 'observability', 'operability', 'orthogonality', 'portability', 'precision', 'predictability', 'process capabilities', 'producibility', 'provability', 'recoverability', 'redundancy', 'relevance', 'reliability', 'repeatability', 'reproducibility', 'resilience', 'responsiveness', 'reusability', 'robustness', 'safety', 'scalability', 'seamlessness', 'securability', 'self-sustainability', 'serviceability', 'simplicity', 'stability', 'standards compliance', 'survivability', 'sustainability', 'tailorability', 'testability', 'timeliness', 'traceability', 'transparency', 'ubiquity', 'understandability', 'upgradability', 'usability', 'vulnerability')
    var_2 = var_1.calver()
    var_3 = var_1.boolean()
    var_4 = module_0.Development()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_4.random).__module__}.{type(var_4.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_4.seed).__module__}.{type(var_4.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    var_5 = var_1.ility()
    var_6 = var_1.__str__()
    assert var_6 == 'Development'
    var_1.validate_enum(var_0, var_0)

def test_case_3():
    var_0 = module_0.Development()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.LICENSES == ['Apache License, 2.0 (Apache-2.0)', 'The BSD 3-Clause License', 'The BSD 2-Clause License', 'GNU General Public License (GPL)', 'General Public License (LGPL)', 'MIT License (MIT)', 'Mozilla Public License 2.0 (MPL-2.0)', 'Common Development and Distribution License (CDDL-1.0)', 'Eclipse Public License (EPL-1.0)']
    assert module_0.OS == ['Arch', 'CentOS', 'Debian', 'Fedora', 'FreeBSD', 'Gentoo', 'Kali', 'Lubuntu', 'Manjaro', 'Mint', 'OS X', 'macOS', 'OpenBSD', 'Slackware', 'Ubuntu', 'Windows 10', 'Windows 7', 'Windows 8', 'Windows 8.1', 'Windows 11', 'elementaryOS', 'macOS', 'openSUSE']
    assert module_0.PROGRAMMING_LANGS == ['ASP', 'Assembly', 'AutoIt', 'Awk', 'Bash', 'C', 'C Shell', 'C#', 'C++', 'Caml', 'Ceylon', 'Clojure', 'CoffeeScript', 'Common Lisp', 'D', 'Dart', 'Delphi', 'Dylan', 'ECMAScript', 'Elixir', 'Emacs Lisp', 'Erlang', 'F#', 'Falcon', 'Fortran', 'GNU Octave', 'Go', 'Groovy', 'Haskell', 'haXe', 'Io', 'J#', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'Lisp', 'Lua', 'Mathematica', 'Objective-C', 'OCaml', 'Perl', 'PHP', 'PL-I', 'PL-SQL', 'PowerShell', 'Prolog', 'Python', 'R', 'Racket', 'Ruby', 'Rust', 'Scala', 'Scheme', 'Smalltalk', 'Tcl', 'Tex', 'Transact-SQL', 'TypeScript', 'Z shell']
    assert module_0.STAGES == ('Pre-alpha', 'Alpha', 'Beta', 'RC', 'Stable')
    assert module_0.SYSTEM_QUALITY_ATTRIBUTES == ('accessibility', 'accountability', 'accuracy', 'adaptability', 'administrability', 'affordability', 'agility', 'auditability', 'autonomy', 'availability', 'compatibility', 'composability', 'confidentiality', 'configurability', 'correctness', 'credibility', 'customizability', 'debuggability', 'degradability', 'demonstrability', 'dependability', 'deployability', 'determinability', 'discoverability', 'distributability', 'durability', 'effectiveness', 'efficiency', 'evolvability', 'extensibility', 'failure transparency', 'fault-tolerance', 'fidelity', 'flexibility', 'inspectability', 'installability', 'integrity', 'interchangeability', 'interoperability', 'learnability', 'localizability', 'maintainability', 'manageability', 'mobility', 'modifiability', 'modularity', 'observability', 'operability', 'orthogonality', 'portability', 'precision', 'predictability', 'process capabilities', 'producibility', 'provability', 'recoverability', 'redundancy', 'relevance', 'reliability', 'repeatability', 'reproducibility', 'resilience', 'responsiveness', 'reusability', 'robustness', 'safety', 'scalability', 'seamlessness', 'securability', 'self-sustainability', 'serviceability', 'simplicity', 'stability', 'standards compliance', 'survivability', 'sustainability', 'tailorability', 'testability', 'timeliness', 'traceability', 'transparency', 'ubiquity', 'understandability', 'upgradability', 'usability', 'vulnerability')
    var_1 = var_0.version()
    var_2 = var_0.programming_language()
    var_3 = var_0.version()
    var_4 = var_0.ility()
    var_5 = var_0.os()

def test_case_4():
    var_0 = module_0.Development()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.providers.development.Development'
    assert f'{type(var_0.random).__module__}.{type(var_0.random).__qualname__}' == 'mimesis.random.Random'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert module_0.LICENSES == ['Apache License, 2.0 (Apache-2.0)', 'The BSD 3-Clause License', 'The BSD 2-Clause License', 'GNU General Public License (GPL)', 'General Public License (LGPL)', 'MIT License (MIT)', 'Mozilla Public License 2.0 (MPL-2.0)', 'Common Development and Distribution License (CDDL-1.0)', 'Eclipse Public License (EPL-1.0)']
    assert module_0.OS == ['Arch', 'CentOS', 'Debian', 'Fedora', 'FreeBSD', 'Gentoo', 'Kali', 'Lubuntu', 'Manjaro', 'Mint', 'OS X', 'macOS', 'OpenBSD', 'Slackware', 'Ubuntu', 'Windows 10', 'Windows 7', 'Windows 8', 'Windows 8.1', 'Windows 11', 'elementaryOS', 'macOS', 'openSUSE']
    assert module_0.PROGRAMMING_LANGS == ['ASP', 'Assembly', 'AutoIt', 'Awk', 'Bash', 'C', 'C Shell', 'C#', 'C++', 'Caml', 'Ceylon', 'Clojure', 'CoffeeScript', 'Common Lisp', 'D', 'Dart', 'Delphi', 'Dylan', 'ECMAScript', 'Elixir', 'Emacs Lisp', 'Erlang', 'F#', 'Falcon', 'Fortran', 'GNU Octave', 'Go', 'Groovy', 'Haskell', 'haXe', 'Io', 'J#', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'Lisp', 'Lua', 'Mathematica', 'Objective-C', 'OCaml', 'Perl', 'PHP', 'PL-I', 'PL-SQL', 'PowerShell', 'Prolog', 'Python', 'R', 'Racket', 'Ruby', 'Rust', 'Scala', 'Scheme', 'Smalltalk', 'Tcl', 'Tex', 'Transact-SQL', 'TypeScript', 'Z shell']
    assert module_0.STAGES == ('Pre-alpha', 'Alpha', 'Beta', 'RC', 'Stable')
    assert module_0.SYSTEM_QUALITY_ATTRIBUTES == ('accessibility', 'accountability', 'accuracy', 'adaptability', 'administrability', 'affordability', 'agility', 'auditability', 'autonomy', 'availability', 'compatibility', 'composability', 'confidentiality', 'configurability', 'correctness', 'credibility', 'customizability', 'debuggability', 'degradability', 'demonstrability', 'dependability', 'deployability', 'determinability', 'discoverability', 'distributability', 'durability', 'effectiveness', 'efficiency', 'evolvability', 'extensibility', 'failure transparency', 'fault-tolerance', 'fidelity', 'flexibility', 'inspectability', 'installability', 'integrity', 'interchangeability', 'interoperability', 'learnability', 'localizability', 'maintainability', 'manageability', 'mobility', 'modifiability', 'modularity', 'observability', 'operability', 'orthogonality', 'portability', 'precision', 'predictability', 'process capabilities', 'producibility', 'provability', 'recoverability', 'redundancy', 'relevance', 'reliability', 'repeatability', 'reproducibility', 'resilience', 'responsiveness', 'reusability', 'robustness', 'safety', 'scalability', 'seamlessness', 'securability', 'self-sustainability', 'serviceability', 'simplicity', 'stability', 'standards compliance', 'survivability', 'sustainability', 'tailorability', 'testability', 'timeliness', 'traceability', 'transparency', 'ubiquity', 'understandability', 'upgradability', 'usability', 'vulnerability')
    var_1 = var_0.ility()
    var_2 = var_0.stage()
    var_3 = 1804
    var_4 = b'@{\xa3'
    var_5 = (var_3, var_4, var_4)
    with pytest.raises(TypeError):
        module_1.BaseProvider(random=var_5)