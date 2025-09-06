#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides analyses for the subject module, based on the module and its AST."""

from __future__ import annotations

import abc
import json
import typing
import functools
import itertools
import dataclasses

from typing import Any
from collections import defaultdict

from pynguin.globl import Globl
from custom_logger import getLogger
from vendor.orderedset import OrderedSet

import pynguin.utils.typetracing as tt

from pynguin.utils import randomness
from pynguin.typesystem import (
    TypeVisitor, Unsupported, TypeInfo,
    ANY, TupleType, UnionType, TypeSystem,
    AnyType, Instance, NoneType, ProperType,
)
from pynguin.runtimevar import RuntimeVariable
from pynguin.instrumentation import CODE_OBJECT_ID_KEY
from pynguin.utils.exceptions import ConstructionFailedException

from pynguin.generic import (
    GenericAccessibleObject,
    GenericCallableAccessibleObject
)


if typing.TYPE_CHECKING:
    from collections.abc import Callable

    import pynguin.algo.archive as arch
    import pynguin.ga.computations as ff

    from pynguin.execution.main import SubjectProperties


LOGGER = getLogger(__name__)


class TestCluster(abc.ABC):
    """Interface for a test cluster."""

    @property
    @abc.abstractmethod
    def type_system(self) -> TypeSystem:
        """Provides the inheritance graph."""

    @property
    @abc.abstractmethod
    def linenos(self) -> int:
        """Provide the number of source code lines."""

    @abc.abstractmethod
    def log_cluster_statistics(self) -> None:
        """Log the signatures of all seen callables."""

    @abc.abstractmethod
    def add_generator(self, generator: GenericAccessibleObject) -> None:
        """Add the given accessible as a generator.

        Args:
            generator: The accessible object
        """

    @abc.abstractmethod
    def add_accessible_object_under_test(
        self, objc: GenericAccessibleObject
    ) -> None:
        """Add accessible object to the objects under test.

        Args:
            objc: The accessible object
            data: The function-description data
        """

    @abc.abstractmethod
    def add_modifier(self, typ: TypeInfo, obj: GenericAccessibleObject) -> None:
        """Add a modifier.

        A modifier is something that can be used to modify the given type,
        for example, a method.

        Args:
            typ: The type that can be modified
            obj: The accessible that can modify
        """

    @property
    @abc.abstractmethod
    def accessible_objects_under_test(self) -> OrderedSet[GenericAccessibleObject]:
        """Provides all accessible objects under test."""

    @property
    @abc.abstractmethod
    def all_accessible_objects(self) -> OrderedSet[GenericAccessibleObject]:
        """Provides all accessible objects.

        Returns:
            The set of all accessible objects
        """

    @abc.abstractmethod
    def num_accessible_objects_under_test(self) -> int:
        """Provide the number of accessible objects under test.

        Useful to check whether there is even something to test.
        """

    @abc.abstractmethod
    def get_generators_for(
        self, typ: ProperType
    ) -> tuple[OrderedSet[GenericAccessibleObject], bool]:
        """Retrieve all known generators for the given type.

        Args:
            typ: The type we want to have the generators for

        Returns:
            The set of all generators for that type, as well as a boolean
              that indicates if all generators have been matched through Any.
              # noqa: DAR202
        """

    @abc.abstractmethod
    def get_modifiers_for(self, typ: ProperType) -> OrderedSet[GenericAccessibleObject]:
        """Get all known modifiers for a type.

        Args:
            typ: The type

        Returns:
            The set of all accessibles that can modify the type  # noqa: DAR202
        """

    @property
    @abc.abstractmethod
    def generators(self) -> dict[ProperType, OrderedSet[GenericAccessibleObject]]:
        """Provides all available generators."""

    @property
    @abc.abstractmethod
    def modifiers(self) -> dict[TypeInfo, OrderedSet[GenericAccessibleObject]]:
        """Provides all available modifiers."""

    @abc.abstractmethod
    def get_random_accessible(self) -> GenericAccessibleObject | None:
        """Provides a random accessible of the unit under test.

        Returns:
            A random accessible, or None if there is none  # noqa: DAR202
        """

    @abc.abstractmethod
    def get_random_call_for(self, typ: ProperType) -> GenericAccessibleObject:
        """Get a random modifier for the given type.

        Args:
            typ: The type

        Returns:
            A random modifier for that type  # noqa: DAR202

        Raises:
            ConstructionFailedException: if no modifiers for the type
                exist# noqa: DAR402
        """

    @abc.abstractmethod
    def get_all_generatable_types(self) -> list[ProperType]:
        """Provides all types that can be generated.

        This includes primitives and collections.

        Returns:
            A list of all types that can be generated  # noqa: DAR202
        """

    @abc.abstractmethod
    def select_concrete_type(self, typ: ProperType) -> ProperType:
        """Select a concrete type from the given type.

        This is required, for example, when handling union types.  Currently, only
        unary types, Any, and Union are handled.

        Args:
            typ: An optional type

        Returns:
            An optional type  # noqa: DAR202
        """

    @abc.abstractmethod
    def track_statistics_values(self, tracking_fun: Callable[[RuntimeVariable, Any], None]) -> None:
        """Track statistics values from the test cluster and its items.

        Args:
            tracking_fun: The tracking function as a callback.
        """

    @abc.abstractmethod
    def update_return_type(
        self, accessible: GenericCallableAccessibleObject, new_type: ProperType
    ) -> None:
        """Update the return for the given accessible to the new seen type.

        Args:
            accessible: the accessible that was observed
            new_type: the new return type
        """

    @abc.abstractmethod
    def update_parameter_knowledge(
        self,
        accessible: GenericCallableAccessibleObject,
        param_name: str,
        knowledge: tt.UsageTraceNode,
    ) -> None:
        """Update the knowledge about the parameter of the given accessible.

        Args:
            accessible: the accessible that was observed.
            param_name: the parameter name for which we have new information.
            knowledge: the new information.
        """

    @abc.abstractmethod
    def promote_object(self, func: GenericAccessibleObject):
        """
        Promotes the object to go into generators/modifiers.

        Args:
            func: function to promote
        """


@dataclasses.dataclass
class SignatureInfo:
    """Another utility class to group information per callable."""

    # A dictionary mapping parameter names and to their developer annotated parameters
    # types.
    # Does not include self, etc.
    annotated_parameter_types: dict[str, str] = dataclasses.field(default_factory=dict)

    # Similar to above, but with guessed parameters types.
    # Contains multiples type guesses.
    guessed_parameter_types: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    # Needed to compute top-n accuracy in the evaluation.
    # Elements are of form (A,B); A is a guess, B is an annotated type.
    # (A,B) is only present, when A is a base type match of B.
    # If it is present, it points to the partial type match between A and B.
    partial_type_matches: dict[str, str] = dataclasses.field(default_factory=dict)

    # Annotated return type, if Any.
    # Does not include constructors.
    annotated_return_type: str | None = None

    # Recorded return type, if Any.
    recorded_return_type: str | None = None


@dataclasses.dataclass
class TypeGuessingStats:
    """Class to gather some type guessing related statistics."""

    # Number of constructors in the MUT.
    number_of_constructors: int = 0

    # Maps names of callables to a signature info object.
    signature_infos: dict[str, SignatureInfo] = dataclasses.field(
        default_factory=lambda: defaultdict(SignatureInfo)
    )


def _serialize_helper(obj):
    """Utility to deal with non-serializable types.

    Args:
        obj: The object to serialize

    Returns:
        A serializable object.
    """
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, SignatureInfo):
        return dataclasses.asdict(obj)
    return obj


class ModuleTestCluster(TestCluster):  # noqa: PLR0904
    """A test cluster for a module.

    Contains all methods/constructors/functions and all required transitive
    dependencies.
    """

    def __init__(self, linenos: int) -> None:  # noqa: D107
        self.__type_system = TypeSystem()
        self.__linenos = linenos
        self.__generators: dict[ProperType, OrderedSet[GenericAccessibleObject]] = defaultdict(
            OrderedSet
        )

        # Modifier belong to a certain class, not type.
        self.__modifiers: dict[TypeInfo, OrderedSet[GenericAccessibleObject]] = defaultdict(
            OrderedSet
        )
        self.__accessible_objects_under_test: OrderedSet[GenericAccessibleObject] = OrderedSet()

    def log_cluster_statistics(self) -> None:  # noqa: D102
        stats = TypeGuessingStats()
        for accessible in self.__accessible_objects_under_test:
            if isinstance(accessible, GenericCallableAccessibleObject):
                accessible.inferred_signature.log_stats_and_guess_signature(
                    accessible.is_constructor(), str(accessible), stats
                )

        Globl.statistics_tracker.track_output_variable(
            RuntimeVariable.SignatureInfos,
            json.dumps(
                stats.signature_infos,
                default=_serialize_helper,
            ),
        )
        Globl.statistics_tracker.track_output_variable(
            RuntimeVariable.NumberOfConstructors,
            str(stats.number_of_constructors),
        )

    def _drop_generator(self, accessible: GenericCallableAccessibleObject):
        gens = self.__generators.get(accessible.generated_type())
        if gens is None:
            return

        gens.discard(accessible)
        if len(gens) == 0:
            self.__generators.pop(accessible.generated_type())

    @staticmethod
    def _add_or_make_union(
        old_type: ProperType, new_type: ProperType, max_size: int = 5
    ) -> UnionType:
        if isinstance(old_type, UnionType):
            items = old_type.items
            if len(items) >= max_size or new_type in items:
                return old_type
            new_type = UnionType(tuple(sorted((*items, new_type))))
        elif old_type in {ANY, new_type}:
            new_type = UnionType((new_type,))
        else:
            new_type = UnionType(tuple(sorted((old_type, new_type))))
        return new_type

    def update_return_type(  # noqa: D102
        self, accessible: GenericCallableAccessibleObject, new_type: ProperType
    ) -> None:
        # Loosely map runtime type to proper type
        old_type = accessible.inferred_signature.return_type

        new_type = self._add_or_make_union(old_type, new_type)
        if old_type == new_type:
            # No change
            return
        self._drop_generator(accessible)
        # Must invalidate entire cache, because subtype relationship might also change
        # the return values which are not new_type or old_type.
        self.get_generators_for.cache_clear()
        self.get_all_generatable_types.cache_clear()
        accessible.inferred_signature.return_type = new_type
        self.__generators[new_type].add(accessible)

    def update_parameter_knowledge(  # noqa: D102
        self,
        accessible: GenericCallableAccessibleObject,
        param_name: str,
        knowledge: tt.UsageTraceNode,
    ) -> None:
        # Store new data
        accessible.inferred_signature.usage_trace[param_name].merge(knowledge)

    @property
    def type_system(self) -> TypeSystem:
        """Provides the type system.

        Returns:
            The type system.
        """
        return self.__type_system

    @property
    def linenos(self) -> int:  # noqa: D102
        return self.__linenos

    def add_generator(self, generator: GenericAccessibleObject) -> None:  # noqa: D102
        generated_type = generator.generated_type()
        # if isinstance(generated_type, NoneType) or generated_type.accept(is_primitive_type):
        #     return
        self.__generators[generated_type].add(generator)

    def add_accessible_object_under_test(  # noqa: D102
        self, objc: GenericAccessibleObject
    ) -> None:
        self.__accessible_objects_under_test.add(objc)

    def add_modifier(  # noqa: D102
        self, typ: TypeInfo, obj: GenericAccessibleObject
    ) -> None:
        self.__modifiers[typ].add(obj)

    @property
    def accessible_objects_under_test(  # noqa: D102
        self,
    ) -> OrderedSet[GenericAccessibleObject]:
        return self.__accessible_objects_under_test
    
    @property
    def all_accessible_objects(self) -> OrderedSet[GenericAccessibleObject]:
        ret_set: OrderedSet[GenericAccessibleObject] = OrderedSet()
        ret_set = ret_set.union(self.__accessible_objects_under_test)
        for vals in self.__modifiers.values():
            ret_set = ret_set.union(vals)
        for vals in self.__generators.values():
            ret_set = ret_set.union(vals)
        return ret_set

    def num_accessible_objects_under_test(self) -> int:  # noqa: D102
        return len(self.__accessible_objects_under_test)

    @functools.lru_cache(maxsize=1024)
    def get_generators_for(  # noqa: D102
        self, typ: ProperType
    ) -> tuple[OrderedSet[GenericAccessibleObject], bool]:
        if isinstance(typ, AnyType):
            # Just take everything when it's Any.
            return (
                OrderedSet(itertools.chain.from_iterable(self.__generators.values())),
                False,
            )

        results: OrderedSet[GenericAccessibleObject] = OrderedSet()
        only_any = True
        for gen_type, generators in self.__generators.items():
            if self.__type_system.is_maybe_subtype(gen_type, typ):
                results.update(generators)
                # Set flag to False as soon as we encounter a generator that is not
                # for Any.
                only_any &= gen_type == ANY

        return results, only_any

    class _FindModifiers(TypeVisitor[OrderedSet[GenericAccessibleObject]]):
        """A visitor to find all modifiers for the given type."""

        def __init__(self, cluster: TestCluster):
            self.cluster = cluster

        def visit_any_type(self, left: AnyType) -> OrderedSet[GenericAccessibleObject]:
            # If it's Any just take everything.
            return OrderedSet(itertools.chain.from_iterable(self.cluster.modifiers.values()))

        def visit_none_type(self, left: NoneType) -> OrderedSet[GenericAccessibleObject]:
            return OrderedSet()

        def visit_instance(self, left: Instance) -> OrderedSet[GenericAccessibleObject]:
            result: OrderedSet[GenericAccessibleObject] = OrderedSet()
            for type_info in self.cluster.type_system.get_superclasses(left.type):
                result.update(self.cluster.modifiers[type_info])
            return result

        def visit_tuple_type(self, left: TupleType) -> OrderedSet[GenericAccessibleObject]:
            return OrderedSet()

        def visit_union_type(self, left: UnionType) -> OrderedSet[GenericAccessibleObject]:
            result: OrderedSet[GenericAccessibleObject] = OrderedSet()
            for element in left.items:
                result.update(element.accept(self))  # type: ignore[arg-type]
            return result

        def visit_unsupported_type(self, left: Unsupported) -> OrderedSet[GenericAccessibleObject]:
            raise NotImplementedError("This type shall not be used during runtime")

    def get_modifiers_for(  # noqa: D102
        self, typ: ProperType
    ) -> OrderedSet[GenericAccessibleObject]:
        return typ.accept(self._FindModifiers(self))

    @property
    def generators(  # noqa: D102
        self,
    ) -> dict[ProperType, OrderedSet[GenericAccessibleObject]]:
        return self.__generators

    @property
    def modifiers(  # noqa: D102
        self,
    ) -> dict[TypeInfo, OrderedSet[GenericAccessibleObject]]:
        return self.__modifiers

    def get_random_accessible(self) -> GenericAccessibleObject | None:  # noqa: D102
        if self.num_accessible_objects_under_test() == 0:
            return None
        return randomness.choice(self.__accessible_objects_under_test)

    def get_random_call_for(  # noqa: D102
        self, typ: ProperType
    ) -> GenericAccessibleObject:
        accessible_objects = self.get_modifiers_for(typ)
        if len(accessible_objects) == 0:
            raise ConstructionFailedException(f"No modifiers for {typ}")
        return randomness.choice(accessible_objects)

    @functools.lru_cache(maxsize=128)
    def get_all_generatable_types(self) -> list[ProperType]:  # noqa: D102
        generatable = OrderedSet(self.__generators.keys())
        generatable.update(self.type_system.primitive_proper_types)
        generatable.update(self.type_system.collection_proper_types)
        return list(generatable)

    def select_concrete_type(self, typ: ProperType) -> ProperType:  # noqa: D102
        if isinstance(typ, AnyType):
            typ = randomness.choice(self.get_all_generatable_types())
        if isinstance(typ, UnionType):
            typ = self.select_concrete_type(randomness.choice(typ.items))
        return typ

    def track_statistics_values(  # noqa: D102
        self, tracking_fun: Callable[[RuntimeVariable, Any], None]
    ) -> None:
        tracking_fun(
            RuntimeVariable.AccessibleObjectsUnderTest,
            self.num_accessible_objects_under_test(),
        )
        tracking_fun(RuntimeVariable.LineNos, self.__linenos)
        tracking_fun(RuntimeVariable.GeneratableTypes, len(self.get_all_generatable_types()))

    def promote_object(self, func: GenericAccessibleObject):
        pass


class ExpandableTestCluster(ModuleTestCluster):
    """A test cluster that keeps track of *all possible* method/constructors/functions
    in the module under test as well as *all* accessible modules under import."""

    def __init__(self, linenos: int) -> None:
        """Create new test cluster."""
        super().__init__(linenos)
        self._backup_accessible_objects: OrderedSet[
            GenericAccessibleObject
        ] = OrderedSet()
        self._all_backups: OrderedSet[GenericAccessibleObject] = OrderedSet()
        self._backup_mode = False

    @property
    def all_accessible_objects(self) -> OrderedSet[GenericAccessibleObject]:
        ret_set = super().all_accessible_objects
        return ret_set.union(self._all_backups)
    
    def set_backup_mode(self, mode: bool):
        """
        Put the test cluster in backup mode, that is, don't add anything to
        generators/modifiers/test cluster, just keep track of the GAOs so they
        can be retrieved later

        Args:
            mode: if True, turn on backup mode

        """
        self._backup_mode = mode

    def get_backup_mode(self) -> bool:
        """
        Returns whether we are currently in backup mode

        Returns:
            the current backup mode
        """
        return self._backup_mode

    def promote_object(self, func: GenericAccessibleObject):
        """
        Promotes the object to go into generators/modifiers.

        Args:
            func: function to promote
        """

        # Otherwise add_generator and add_modifier will do nothing
        assert self._backup_mode is False
        assert isinstance(func, GenericCallableAccessibleObject)

        if func not in self._backup_accessible_objects: return

        # To prevent recursion when adding dependencies, remove this
        # from backup objects.
        self._backup_accessible_objects.remove(func)
        
        # Add it as a generator if it can generate types
        self.add_generator(func)

        # Add it as a modifier if it is a method
        if func.is_method():
            modified_type = func.owner
            assert modified_type is not None
            self.add_modifier(modified_type, func)

        # Promote any types in the type signature to the test cluster
        signature = func.inferred_signature
        for _, type_ in signature.original_parameters.items():
            types = type_.items if isinstance(type_, UnionType) else {type_}

            constructors = [
                obj for obj in self._backup_accessible_objects
                if obj.is_constructor() and obj.owner in types
            ]
            for obj in constructors: self.promote_object(obj)

        # Also retrieve all the methods for a constructor
        if func.is_constructor():
            type_under_test = func.owner
            assert type_under_test is not None

            methods = [
                obj for obj in self._backup_accessible_objects
                if obj.is_method() and obj.owner == type_under_test
            ]
            for obj in methods: self.promote_object(obj)

    def add_generator(self, generator: GenericAccessibleObject) -> None:
        """Add the given accessible as a generator.

        Args:
            generator: The accessible object
        """
        if not self._backup_mode:
            super().add_generator(generator)
        else:
            self._backup_accessible_objects.add(generator)
            self._all_backups.add(generator)

    def add_accessible_object_under_test(self, obj: GenericAccessibleObject) -> None:
        """Add accessible object to the objects under test.

        Args:
            obj: The accessible object
        """
        if not self._backup_mode:
            super().add_accessible_object_under_test(obj)
        else:
            self._backup_accessible_objects.add(obj)
            self._all_backups.add(obj)

    def add_modifier(self, type_: type, obj: GenericAccessibleObject) -> None:
        """Add a modifier.

        A modified is something that can be used to modify
        the given type, e.g. a method.

        Args:
            type_: The type that can be modified
            obj: The accessible that can modify
        """
        if not self._backup_mode:
            super().add_modifier(type_, obj)
        else:
            self._backup_accessible_objects.add(obj)
            self._all_backups.add(obj)

    def was_added_in_backup(self, obj: GenericAccessibleObject):
        """Returns true if the object `obj` was added as a backup. For statistics
        tracking purposes.

        Args:
            obj: the object to check

        Returns:
            True if obj was added in backup mode
        """
        return obj in self._all_backups


class FilteredModuleTestCluster(TestCluster):  # noqa: PLR0904
    """A test cluster wrapping another test cluster.

    Delegates most methods to the wrapped delegate.  This cluster filters out
    accessible objects under test that are already fully covered, in order to focus
    the search on areas that are not yet fully covered.
    """

    @property
    def type_system(self) -> TypeSystem:  # noqa: D102
        return self.__delegate.type_system

    def update_return_type(  # noqa: D102
        self, accessible: GenericCallableAccessibleObject, new_type: ProperType
    ) -> None:
        self.__delegate.update_return_type(accessible, new_type)

    def update_parameter_knowledge(  # noqa: D102
        self,
        accessible: GenericCallableAccessibleObject,
        param_name: str,
        knowledge: tt.UsageTraceNode,
    ) -> None:
        self.__delegate.update_parameter_knowledge(accessible, param_name, knowledge)

    @property
    def linenos(self) -> int:  # noqa: D102
        return self.__delegate.linenos

    def log_cluster_statistics(self) -> None:  # noqa: D102
        self.__delegate.log_cluster_statistics()

    def add_generator(self, generator: GenericAccessibleObject) -> None:  # noqa: D102
        self.__delegate.add_generator(generator)

    def add_accessible_object_under_test(  # noqa: D102
        self, objc: GenericAccessibleObject
    ) -> None:
        self.__delegate.add_accessible_object_under_test(objc)

    def add_modifier(  # noqa: D102
        self, typ: TypeInfo, obj: GenericAccessibleObject
    ) -> None:
        self.__delegate.add_modifier(typ, obj)

    def track_statistics_values(  # noqa: D102
        self, tracking_fun: Callable[[RuntimeVariable, Any], None]
    ) -> None:
        self.__delegate.track_statistics_values(tracking_fun)

    def __init__(  # noqa: D107
        self,
        delegate: TestCluster,
        archive: arch.Archive,
        subject_properties: SubjectProperties,
        targets: OrderedSet[ff.TestCaseFitnessFunction],
    ) -> None:
        self.__delegate = delegate
        self.__subject_properties = subject_properties
        self.__code_object_id_to_accessible_objects: dict[int, GenericCallableAccessibleObject] = {
            json.loads(acc.callable.__code__.co_consts[0])[CODE_OBJECT_ID_KEY]: acc
            for acc in delegate.accessible_objects_under_test
            if isinstance(acc, GenericCallableAccessibleObject)
            and hasattr(acc.callable, "__code__")
        }
        # Checking for __code__ is necessary, because the __init__ of a class that
        # does not define __init__ points to some internal CPython stuff.

        self.__accessible_to_targets: dict[GenericCallableAccessibleObject, OrderedSet] = {
            acc: OrderedSet() for acc in self.__code_object_id_to_accessible_objects.values()
        }
        for target in targets:
            if (acc := self.__get_accessible_object_for_target(target)) is not None:
                targets_for_acc = self.__accessible_to_targets[acc]
                targets_for_acc.add(target)

        # Get informed by archive when a target is covered
        archive.add_on_target_covered(self.on_target_covered)

    def __get_accessible_object_for_target(
        self, target: ff.TestCaseFitnessFunction
    ) -> GenericCallableAccessibleObject | None:
        code_object_id: int | None = target.code_object_id
        while code_object_id is not None:
            if (
                acc := self.__code_object_id_to_accessible_objects.get(code_object_id, None)
            ) is not None:
                return acc
            code_object_id = self.__subject_properties.existing_code_objects[
                code_object_id
            ].parent_code_object_id
        return None

    def on_target_covered(self, target: ff.TestCaseFitnessFunction) -> None:
        """A callback function to get informed by an archive when a target is covered.

        Args:
            target: The newly covered target
        """
        acc = self.__get_accessible_object_for_target(target)
        if acc is not None:
            targets_for_acc = self.__accessible_to_targets.get(acc)
            assert targets_for_acc is not None
            targets_for_acc.remove(target)
            if len(targets_for_acc) == 0:
                self.__accessible_to_targets.pop(acc)
                LOGGER.debug(
                    "Removed %s from test cluster because all targets within it have been covered.",
                    acc,
                )

    @property
    def accessible_objects_under_test(  # noqa: D102
        self,
    ) -> OrderedSet[GenericAccessibleObject]:
        accessibles = self.__accessible_to_targets.keys()
        if len(accessibles) == 0:
            # Should never happen, just in case everything is already covered?
            return self.__delegate.accessible_objects_under_test
        return OrderedSet(accessibles)
    
    @property
    def all_accessible_objects(self) -> OrderedSet[GenericAccessibleObject]:
        # TODO(ANON): These are not filtered
        ret_set: OrderedSet[GenericAccessibleObject] = OrderedSet()
        ret_set = ret_set.union(self.accessible_objects_under_test)
        for vals in self.modifiers.values():
            ret_set = ret_set.union(vals)
        for vals in self.generators.values():
            ret_set = ret_set.union(vals)
        return ret_set

    def num_accessible_objects_under_test(self) -> int:  # noqa: D102
        return self.__delegate.num_accessible_objects_under_test()

    def get_generators_for(  # noqa: D102
        self, typ: ProperType
    ) -> tuple[OrderedSet[GenericAccessibleObject], bool]:
        return self.__delegate.get_generators_for(typ)

    def get_modifiers_for(  # noqa: D102
        self, typ: ProperType
    ) -> OrderedSet[GenericAccessibleObject]:
        return self.__delegate.get_modifiers_for(typ)

    @property
    def generators(  # noqa: D102
        self,
    ) -> dict[ProperType, OrderedSet[GenericAccessibleObject]]:
        return self.__delegate.generators

    @property
    def modifiers(  # noqa: D102
        self,
    ) -> dict[TypeInfo, OrderedSet[GenericAccessibleObject]]:
        return self.__delegate.modifiers

    def get_random_accessible(self) -> GenericAccessibleObject | None:  # noqa: D102
        accessibles = self.__accessible_to_targets.keys()
        if len(accessibles) == 0:
            return self.__delegate.get_random_accessible()
        return randomness.choice(OrderedSet(accessibles))

    def get_random_call_for(  # noqa: D102
        self, typ: ProperType
    ) -> GenericAccessibleObject:
        return self.__delegate.get_random_call_for(typ)

    def get_all_generatable_types(self) -> list[ProperType]:  # noqa: D102
        return self.__delegate.get_all_generatable_types()

    def select_concrete_type(self, typ: ProperType) -> ProperType:  # noqa: D102
        return self.__delegate.select_concrete_type(typ)

    def promote_object(self, func: GenericAccessibleObject):
        return self.__delegate.promote_object(func)
