import enum
import queue
import astroid
import inspect
import builtins
import importlib
import dataclasses

from typing import Any
from pathlib import Path
from types import (
    ModuleType, WrapperDescriptorType,
    GenericAlias, BuiltinFunctionType,
    FunctionType, MethodDescriptorType,
)
from vendor.custom_logger import getLogger

from pynguin.syntaxtree import (
    get_class_node_from_ast,
    get_function_description,
    get_function_node_from_ast
)

from pynguin.generic import (
    GenericMethod, GenericFunction,
    GenericEnum, GenericConstructor,
)

from pynguin.setup.testcluster import (
    ModuleTestCluster,
    ExpandableTestCluster
)

from pynguin.globl import Globl
from pynguin.typesystem.main import TypeInfo
from pynguin.config import TypeInferenceStrategy
from pynguin.utils.type_utils import COLLECTIONS, PRIMITIVES
from pynguin.utils.type_utils import get_class_that_defined_method

LOGGER = getLogger(__name__)


# A set of modules that shall be blacklisted from analysis (keep them sorted to ease
# future manipulations or looking up module names of this set!!!):
# The modules that are listed here are not prohibited from execution, but Pynguin will
# not consider any classes or functions from these modules for generating inputs to
# other routines
MODULE_BLACKLIST = frozenset((
    "__future__",
    "_frozen_importlib",
    "_thread",
    "abc",
    "argparse",
    "asyncio",
    "atexit",
    "builtins",
    "cmd",
    "code",
    "codeop",
    "collections.abc",
    "compileall",
    "concurrent",
    "concurrent.futures",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "csv",
    "ctypes",
    "dbm",
    "dis",
    "filecmp",
    "fileinput",
    "fnmatch",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "glob",
    "importlib",
    "io",
    "itertools",
    "linecache",
    "logging",
    "logging.config",
    "logging.handlers",
    "marshal",
    "mmap",
    "multiprocessing",
    "multiprocessing.shared_memory",
    "netrc",
    "operator",
    "os",
    "os.path",
    "pathlib",
    "pickle",
    "pickletools",
    "plistlib",
    "py_compile",
    "queue",
    "random",
    "reprlib",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shutil",
    "signal",
    "six",  # Not from STDLIB
    "socket",
    "sre_compile",
    "sre_parse",
    "ssl",
    "stat",
    "subprocess",
    "sys",
    "tarfile",
    "tempfile",
    "threading",
    "timeit",
    "trace",
    "traceback",
    "tracemalloc",
    "types",
    "typing",
    "warnings",
    "weakref",
))

# Blacklist for methods.
METHOD_BLACKLIST = frozenset(("time.sleep",))

def _is_blacklisted(element: Any) -> bool:
    """Checks if the given element belongs to the blacklist.

    Args:
        element: The element to check

    Returns:
        Is the element blacklisted?
    """
    module_blacklist = set(MODULE_BLACKLIST).union(Globl.conf.ignore_modules)
    method_blacklist = set(METHOD_BLACKLIST).union(Globl.conf.ignore_methods)

    if inspect.ismodule(element):
        return element.__name__ in module_blacklist
    if inspect.isclass(element):
        if element.__module__ == "builtins" and (element in PRIMITIVES or element in COLLECTIONS):
            # Allow some builtin types
            return False
        return element.__module__ in module_blacklist
    if inspect.isfunction(element):
        # Some modules can be run standalone using a main function or provide a small
        # set of tests ('test'). We don't want to include those functions.
        return (
            element.__module__ in module_blacklist
            or element.__qualname__.startswith((
                "main",
                "test",
            ))
            or f"{element.__module__}.{element.__qualname__}" in method_blacklist
        )
    # Something that is not supported yet.
    return False


@dataclasses.dataclass
class _ModuleParseResult:
    """A data wrapper for an imported and parsed module."""

    linenos: int
    module_name: str
    module: ModuleType
    syntax_tree: astroid.Module | None


def import_module(module_name: str) -> ModuleType:
    """Imports a module by name.

    Unlike the built-in :py:func:`importlib.import_module`, this function also supports
    importing module aliases.

    Args:
        module_name: The fully-qualified name of the module

    Returns:
        The imported module
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        try:
            package_name, submodule_name = module_name.rsplit(".", 1)
        except ValueError as e:
            raise error from e

        try:
            package = import_module(package_name)
        except ModuleNotFoundError as e:
            raise error from e

        try:
            submodule = getattr(package, submodule_name)
        except AttributeError as e:
            raise error from e

        if not inspect.ismodule(submodule):
            raise error

        return submodule


def parse_module(module_name: str) -> _ModuleParseResult:
    """Parses a module and extracts its module-type and AST.

    If the source code is not available it is not possible to build an AST.  In this
    case the respective field of the :py:class:`_ModuleParseResult` will contain the
    value ``None``.  This is the case, for example, for modules written in native code,
    for example, in C.

    Args:
        module_name: The fully-qualified name of the module

    Returns:
        A tuple of the imported module type and its optional AST
    """
    module = import_module(module_name)
    syntax_tree: astroid.Module | None = None
    linenos: int = -1
    try:
        source_file = inspect.getsourcefile(module)
        source_code = inspect.getsource(module)
        syntax_tree = astroid.parse(
            code=source_code,
            module_name=module_name,
            path=source_file if source_file is not None else "",
        )
        linenos = len(source_code.splitlines())

    except (TypeError, OSError) as error:
        LOGGER.debug(
            f"Could not retrieve source code for module {module_name} "  # noqa: G004
            f"({error}). "
            f"Cannot derive syntax tree to allow Pynguin using more precise analysis."
        )

    return _ModuleParseResult(
        linenos=linenos,
        module_name=module_name,
        module=module,
        syntax_tree=syntax_tree,
    )


class _ParseResults(dict):  # noqa: FURB189
    def __missing__(self, key):
        # Parse module on demand
        res = self[key] = parse_module(key)
        return res


def __resolve_dependencies(
    root_module: _ModuleParseResult,
    type_inference_strategy: TypeInferenceStrategy,
    test_cluster: ModuleTestCluster | ExpandableTestCluster,
    make_expandable: bool,
    max_cluster_recursion: int
) -> None:
    parse_results: dict[str, _ModuleParseResult] = _ParseResults()
    parse_results[root_module.module_name] = root_module

    # Provide a set of seen modules, classes and functions for fixed-point iteration
    seen_modules: set[ModuleType] = set()
    seen_classes: set[Any] = set()
    seen_functions: set[Any] = set()

    # Always analyse builtins
    __analyse_included_classes(
        module=builtins,
        root_module_name=root_module.module_name,
        type_inference_strategy=type_inference_strategy,
        test_cluster=test_cluster,
        seen_classes=seen_classes,
        parse_results=parse_results
    )
    test_cluster.type_system.enable_numeric_tower()

    # Start with root module, i.e., the module under test.
    wait_list: queue.SimpleQueue[ModuleType] = queue.SimpleQueue()
    wait_list.put(root_module.module)

    while not wait_list.empty():
        current_module = wait_list.get()
        if current_module in seen_modules:
            # Skip the module, we have already analysed it before
            continue
        if _is_blacklisted(current_module):
            # Don't include anything from the blacklist
            continue

        # Analyze all classes and functions found in the current module
        __analyse_included_classes_and_functions(
            module=current_module,
            root_module_name=root_module.module_name,
            type_inference_strategy=type_inference_strategy,
            test_cluster=test_cluster,
            seen_classes=seen_classes,
            seen_functions=seen_functions,
            parse_results=parse_results,
        )

        # Collect the modules that are included by this module and add
        # them for further processing.
        for included_module in filter(inspect.ismodule, vars(current_module).values()):
            wait_list.put(included_module)

        # Take care that we know for future iterations that we have already analysed
        # this module before
        seen_modules.add(current_module)

    # If we're making an expandable cluster, create the backup set of GAOs.
    # Four main differences from the code above:
    #   1. Test cluster's backup mode = True (by defaults)
    #      (unless we're making whole expandable cluster from start)
    #   2. Analyse modules with recursion <= max_cluster_recursion
    #   3. Bypass any blacklist (skip any _is_blacklisted call)
    #   4. Also analyze dependencies of imported objects' modules
    
    if make_expandable:
        test_cluster.set_backup_mode(not Globl.seeding_conf.expand_cluster)

        seen_modules = set()
        wait_list: list[list[ModuleType]] = [
            [] for _ in range(max_cluster_recursion + 1)
        ]
        wait_list[0].append(root_module.module)

        def should_analyse_module(module: ModuleType):
            return (
                hasattr(module, '__file__')
                and module.__file__.startswith(Globl.project_path)
                and module not in seen_modules
            )

        for level in range(max_cluster_recursion + 1):
            for current_module in wait_list[level]:
                if current_module in seen_modules:
                    continue

                __analyse_included_classes_and_functions(
                    module=current_module,
                    root_module_name=root_module.module_name,
                    type_inference_strategy=type_inference_strategy,
                    test_cluster=test_cluster,
                    seen_classes=seen_classes,
                    seen_functions=seen_functions,
                    parse_results=parse_results,
                    bypass_blacklist=True
                )

                seen_modules.add(current_module)

                if level == max_cluster_recursion:
                    # Skip analysing later dependencies
                    continue

                for included_module in filter(
                    should_analyse_module,
                    vars(current_module).values()
                ):
                    wait_list[level+1].append(included_module)

                # also include modules of included classes and functions
                for obj in filter(
                    lambda x: hasattr(x, '__module__')
                    and isinstance(x.__module__, str),
                    vars(current_module).values()
                ):
                    try:
                        included_module = import_module(obj.__module__)
                    except ModuleNotFoundError:
                        LOGGER.info("Failed to import module '%s'. Skipping...", obj.__module__)
                        continue
                    if should_analyse_module(included_module):
                        wait_list[level+1].append(included_module)    
        
        test_cluster.set_backup_mode(False)

    LOGGER.info("Analyzed project to create test cluster")
    LOGGER.info("Modules:   %5i", len(seen_modules))
    LOGGER.info("Functions: %5i", len(seen_functions))
    LOGGER.info("Classes:   %5i", len(seen_classes))

    test_cluster.type_system.push_attributes_down()


def __is_constructor(method_name: str) -> bool:
    return method_name == "__init__"


def __is_protected(method_name: str) -> bool:
    return method_name.startswith("_") and not method_name.startswith("__")


def __is_private(method_name: str) -> bool:
    return method_name.startswith("__") and not method_name.endswith("__")


def __is_method_defined_in_class(class_: type, method: object) -> bool:
    return class_ == get_class_that_defined_method(method)


def __analyse_function(
    *,
    func_name: str,
    func: FunctionType,
    type_inference_strategy: TypeInferenceStrategy,
    module_tree: astroid.Module | None,
    test_cluster: ModuleTestCluster,
    add_to_test: bool,
) -> None:
    if __is_private(func_name) or __is_protected(func_name):
        LOGGER.debug("Skipping function %s from analysis", func_name)
        return
    if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
        if add_to_test:
            raise ValueError("Pynguin cannot handle Coroutine in SUT. Stopping.")
        # Coroutine outside the SUT are not problematic, just exclude them.
        LOGGER.debug("Skipping coroutine %s outside of SUT", func_name)
        return

    LOGGER.debug("Analysing function %s", func_name)
    inferred_signature = test_cluster.type_system.infer_type_info(
        func,
        type_inference_strategy=type_inference_strategy,
    )
    func_ast = get_function_node_from_ast(module_tree, func_name)
    description = get_function_description(func_ast)
    raised_exceptions = description.raises if description is not None else set()
    generic_function = GenericFunction(func, inferred_signature, raised_exceptions, func_name)
    test_cluster.add_generator(generic_function)
    if add_to_test:
        test_cluster.add_accessible_object_under_test(generic_function)


def __analyse_class(
    *,
    type_info: TypeInfo,
    type_inference_strategy: TypeInferenceStrategy,
    module_tree: astroid.Module | None,
    test_cluster: ModuleTestCluster,
    add_to_test: bool,
) -> None:
    LOGGER.debug("Analysing class %s", type_info)
    class_ast = get_class_node_from_ast(module_tree, type_info.name)
    __add_symbols(class_ast, type_info)
    if type_info.raw_type is tuple:
        # Tuple is problematic...
        return

    constructor_ast = get_function_node_from_ast(class_ast, "__init__")
    description = get_function_description(constructor_ast)
    raised_exceptions = description.raises if description is not None else set()

    if issubclass(type_info.raw_type, enum.Enum):
        generic: GenericEnum | GenericConstructor = GenericEnum(type_info)
        if isinstance(generic, GenericEnum) and len(generic.names) == 0:
            LOGGER.debug(
                "Skipping enum %s from test cluster, it has no fields.",
                type_info.full_name,
            )
            return
    else:
        generic = GenericConstructor(
            type_info,
            test_cluster.type_system.infer_type_info(
                type_info.raw_type.__init__,
                type_inference_strategy=type_inference_strategy,
            ),
            raised_exceptions,
        )
        generic.inferred_signature.return_type = test_cluster.type_system.convert_type_hint(
            type_info.raw_type
        )

    if not (
        type_info.is_abstract
        or type_info.raw_type in COLLECTIONS
        or type_info.raw_type in PRIMITIVES
    ):
        # Don't add constructors for abstract classes and for builtins. We generate
        # the latter ourselves.
        test_cluster.add_generator(generic)
        if add_to_test:
            test_cluster.add_accessible_object_under_test(generic)

    for method_name, method in inspect.getmembers(type_info.raw_type, inspect.isfunction):
        __analyse_method(
            type_info=type_info,
            method_name=method_name,
            method=method,
            type_inference_strategy=type_inference_strategy,
            class_tree=class_ast,
            test_cluster=test_cluster,
            add_to_test=add_to_test,
        )


# Some symbols are not interesting for us.
IGNORED_SYMBOLS: set[str] = {
    "__new__",
    "__init__",
    "__repr__",
    "__str__",
    "__sizeof__",
    "__getattribute__",
    "__getattr__",
}


def __add_symbols(class_ast: astroid.ClassDef | None, type_info: TypeInfo) -> None:
    """Tries to infer what symbols can be found on an instance of the given class.

    We also try to infer what attributes are defined in '__init__'.

    Args:
        class_ast: The AST Node of the class.
        type_info: The type info.
    """
    if class_ast is not None:
        type_info.instance_attributes.update(tuple(class_ast.instance_attrs))
    type_info.attributes.update(type_info.instance_attributes)
    type_info.attributes.update(tuple(vars(type_info.raw_type)))
    type_info.attributes.difference_update(IGNORED_SYMBOLS)


def __analyse_method(
    *,
    type_info: TypeInfo,
    method_name: str,
    method: (FunctionType | BuiltinFunctionType | WrapperDescriptorType | MethodDescriptorType),
    type_inference_strategy: TypeInferenceStrategy,
    class_tree: astroid.ClassDef | None,
    test_cluster: ModuleTestCluster,
    add_to_test: bool,
) -> None:
    if (
        __is_private(method_name)
        or __is_protected(method_name)
        or __is_constructor(method_name)
        or not __is_method_defined_in_class(type_info.raw_type, method)
    ):
        LOGGER.debug("Skipping method %s from analysis", method_name)
        return
    if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
        if add_to_test:
            raise ValueError("Pynguin cannot handle Coroutine in SUT. Stopping.")
        # Coroutine outside the SUT are not problematic, just exclude them.
        LOGGER.debug("Skipping coroutine %s outside of SUT", method_name)
        return

    LOGGER.debug("Analysing method %s.%s", type_info.full_name, method_name)
    inferred_signature = test_cluster.type_system.infer_type_info(
        method,
        type_inference_strategy=type_inference_strategy,
    )
    method_ast = get_function_node_from_ast(class_tree, method_name)
    description = get_function_description(method_ast)
    raised_exceptions = description.raises if description is not None else set()
    generic_method = GenericMethod(
        type_info, method, inferred_signature, raised_exceptions, method_name
    )
    test_cluster.add_generator(generic_method)
    test_cluster.add_modifier(type_info, generic_method)
    if add_to_test:
        test_cluster.add_accessible_object_under_test(generic_method)


def __analyse_included_classes_and_functions(
    *,
    module: ModuleType,
    root_module_name: str,
    type_inference_strategy: TypeInferenceStrategy,
    test_cluster: ModuleTestCluster,
    parse_results: dict[str, _ModuleParseResult],
    seen_classes: set[type],
    seen_functions: set,
    bypass_blacklist: bool = False
):
    """Sequentially calls
    :py:func:`__analyse_included_classes` and
    :py:func:`__analyse_included_functions`.
    """
    __analyse_included_classes(
        module=module,
        root_module_name=root_module_name,
        type_inference_strategy=type_inference_strategy,
        test_cluster=test_cluster,
        parse_results=parse_results,
        seen_classes=seen_classes,
        bypass_blacklist=bypass_blacklist
    )
    __analyse_included_functions(
        module=module,
        root_module_name=root_module_name,
        type_inference_strategy=type_inference_strategy,
        test_cluster=test_cluster,
        parse_results=parse_results,
        seen_functions=seen_functions,
        bypass_blacklist=bypass_blacklist
    )


def __analyse_included_classes(
    *,
    module: ModuleType,
    root_module_name: str,
    type_inference_strategy: TypeInferenceStrategy,
    test_cluster: ModuleTestCluster,
    parse_results: dict[str, _ModuleParseResult],
    seen_classes: set[type],
    bypass_blacklist: bool = False
) -> None:
    work_list = list(
        filter(
            lambda x: inspect.isclass(x)
            and (
                bypass_blacklist
                or not _is_blacklisted(x)
            ),
            vars(module).values(),
        )
    )

    # TODO(fk) inner classes?
    while len(work_list) > 0:
        current = work_list.pop(0)
        if current in seen_classes:
            continue
        seen_classes.add(current)

        type_info = test_cluster.type_system.to_type_info(current)

        # Skip if the class is _ObjectProxyMethods, as it is broken
        # since __module__ is not well defined on it.
        if isinstance(current.__module__, property):
            LOGGER.info("Skipping class that has a property __module__: %s", current)
            continue

        # Skip some C-extension modules that are not publicly accessible.
        try:
            results = parse_results[current.__module__]
        except ModuleNotFoundError as error:
            if getattr(current, "__file__", None) is None or Path(current.__file__).suffix in {
                ".so",
                ".pyd",
            }:
                LOGGER.info("C-extension module not found: %s", current.__module__)
                continue
            raise error
        except Exception as e:
            LOGGER.warning(
                "Failed to get module_tree of '%s' under module '%s'",
                current, current.__module__
            )
        
        add_to_test = current.__module__ == root_module_name

        __analyse_class(
            type_info=type_info,
            type_inference_strategy=type_inference_strategy,
            module_tree=results.syntax_tree,
            test_cluster=test_cluster,
            add_to_test=add_to_test,
        )

        if hasattr(current, "__bases__"):
            for base in current.__bases__:
                # TODO(fk) base might be an instance.
                #  Ignored for now.
                #  Probably store Instance in graph instead of TypeInfo?
                if isinstance(base, GenericAlias):
                    base = base.__origin__  # noqa: PLW2901

                base_info = test_cluster.type_system.to_type_info(base)
                test_cluster.type_system.add_subclass_edge(
                    super_class=base_info, sub_class=type_info
                )
                work_list.append(base)


def __analyse_included_functions(
    *,
    module: ModuleType,
    root_module_name: str,
    type_inference_strategy: TypeInferenceStrategy,
    test_cluster: ModuleTestCluster,
    parse_results: dict[str, _ModuleParseResult],
    seen_functions: set,
    bypass_blacklist: bool = False
) -> None:
    for current in filter(
        lambda x: inspect.isfunction(x)
        and x.__name__ != '<lambda>'
        and (
            bypass_blacklist
            or not _is_blacklisted(x)
        ),
        vars(module).values(),
    ):
        if current in seen_functions:
            continue
        seen_functions.add(current)

        add_to_test = current.__module__ == root_module_name

        try:
            module_tree = parse_results[current.__module__].syntax_tree
        except Exception as e:
            LOGGER.warning(
                "Failed to get module_tree of '%s' under module '%s'",
                current, current.__module__
            )

        __analyse_function(
            func_name=current.__qualname__,
            func=current,
            type_inference_strategy=type_inference_strategy,
            module_tree=module_tree,
            test_cluster=test_cluster,
            add_to_test=add_to_test,
        )


def analyse_module(
    parsed_module: _ModuleParseResult,
    type_inference_strategy: TypeInferenceStrategy = TypeInferenceStrategy.TYPE_HINTS,
) -> ModuleTestCluster:
    """Analyses a module to build a test cluster.

    Args:
        parsed_module: The parsed module
        type_inference_strategy: The type inference strategy to use.

    Returns:
        A test cluster for the module
    """
    make_expandable = (
        Globl.seeding_conf.expand_cluster or
        Globl.seeding_conf.allow_expandable_cluster
    )
    max_cluster_recursion = Globl.seeding_conf.max_cluster_recursion
    if make_expandable:
        test_cluster = ExpandableTestCluster(linenos=parsed_module.linenos)
    else:
        test_cluster = ModuleTestCluster(linenos=parsed_module.linenos)
    LOGGER.info(f"Setting up test cluster with {test_cluster.__class__.__name__}")

    __resolve_dependencies(
        root_module=parsed_module,
        type_inference_strategy=type_inference_strategy,
        test_cluster=test_cluster,
        make_expandable=make_expandable,
        max_cluster_recursion=max_cluster_recursion
    )
    return test_cluster


def generate_test_cluster() -> ModuleTestCluster:
    """Generates a new test cluster from the given module.

    Returns:
        A new test cluster for the given module
    """

    type_inference_strategy = Globl.conf.type_inference.type_inference_strategy
    return analyse_module(parse_module(Globl.module_name), type_inference_strategy)