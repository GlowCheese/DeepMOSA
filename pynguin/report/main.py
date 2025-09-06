#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides capabilities to create a coverage report."""

from __future__ import annotations
import pynguin.__version__ as ver

import dataclasses
import datetime
import importlib.resources
import inspect
import sys
import typing
import xml.etree.ElementTree as ET  # noqa: S405

import pygments

from typing import Literal
from jinja2 import Template
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonLexer

from pynguin.globl import Globl
from pynguin.config import CoverageMetric
import pynguin.ga.computations as ff

from vendor.orderedset import OrderedSet


if typing.TYPE_CHECKING:
    from pathlib import Path

    import pynguin.ga.testsuitechromosome as tsc
    from pynguin.execution import ExecutionTracer


@dataclasses.dataclass(frozen=True)
class _CoverageEntry:
    """How many things exist and how many are covered?"""

    covered: int = 0
    existing: int = 0

    def __add__(self, other: _CoverageEntry) -> _CoverageEntry:
        """Add data from another coverage entry to this one.

        Args:
            other: another CoverageEntry whose values are added to this one.

        Returns:
            A new coverage entry with the summed up elements of self and other.
        """
        return _CoverageEntry(self.covered + other.covered, self.existing + other.existing)


@dataclasses.dataclass
class _LineAnnotation:
    """Coverage information for a single line."""

    line_no: int

    total: _CoverageEntry

    branches: _CoverageEntry

    branchless_code_objects: _CoverageEntry

    lines: _CoverageEntry

    def message(self):
        """Compute the message that should be displayed as a tool tip.

        Returns:
            The message for this line.
        """
        msgs = []
        if self.branches.existing > 0:
            msgs.append(f"{self.branches.covered}/{self.branches.existing} branches covered")
        if self.branchless_code_objects.existing > 0:
            msgs.append(
                f"{self.branchless_code_objects.covered}/"
                f"{self.branchless_code_objects.existing}"
                f" branchless code objects covered"
            )
        if self.lines.existing > 0:
            # No need to say something like "1 out of 1 lines executed".
            msgs.append(f"Line {self.line_no}{'' if self.lines.covered == 1 else ' not'} covered")
        return "; ".join(msgs)

    def __add__(self, other: _LineAnnotation) -> _LineAnnotation:
        """Add data from another line annotation entry to this one.

        Args:
            other: another line annotation whose values are added to this one.

        Returns:
            A new line annotation with the summed up elements of self and other.
        """
        assert self.line_no == other.line_no
        return _LineAnnotation(
            self.line_no,
            self.total + other.total,
            self.branches + other.branches,
            self.branchless_code_objects + other.branchless_code_objects,
            self.lines + other.lines,
        )


@dataclasses.dataclass
class CoverageReport:
    """All coverage related data required to create a coverage report."""

    module: str

    # Raw source code of the module under test
    source: list[str]

    # Information about total covered branches
    branches: _CoverageEntry

    # Information about total covered branchless code objects
    branchless_code_objects: _CoverageEntry

    # Information about total covered lines
    lines: _CoverageEntry

    # Coverage information per line
    line_annotations: list[_LineAnnotation]

    # Achieved branch coverage
    branch_coverage: float | None = None

    # Achieved line coverage
    line_coverage: float | None = None


@dataclasses.dataclass
class CoverageReportJson:
    num_lines: int
    line_coverage: float
    branch_coverage: float
    empty_lines: list[int]
    covered_lines: list[int]
    branches: dict[str, dict[Literal['covered', 'total'], int]]

    @staticmethod
    def make_from(cov_report: CoverageReport) -> "CoverageReportJson":
        num_line = len(cov_report.line_annotations)
        line_coverage = cov_report.line_coverage
        branch_coverage = cov_report.branch_coverage
        empty_lines = [
            line.line_no for line in cov_report.line_annotations
            if line.lines.existing == 0
        ]
        covered_lines = [
            line.line_no for line in cov_report.line_annotations
            if line.lines.covered == 1
        ]
        branches = {
            str(line.line_no): {
                "covered": line.branches.covered,
                "total": line.branches.existing
            }
            for line in cov_report.line_annotations
            if line.branches.existing > 0
        }
        return CoverageReportJson(
            num_line, line_coverage, branch_coverage,
            empty_lines, covered_lines, branches
        )
    
    @property
    def num_covered_lines(self):
        return len(self.covered_lines)
    
    @property
    def num_covered_branches(self):
        return sum(v['covered'] for v in self.branches.values())
    
    @property
    def num_branches(self):
        return sum(v['total'] for v in self.branches.values())
    
    def only_include_lines_between(self, lineno, end_lineno):
        self.num_lines = end_lineno - lineno + 1
        self.empty_lines = [
            line for line in self.empty_lines
            if lineno <= line <= end_lineno
        ]
        self.covered_lines = [
            line for line in self.covered_lines
            if lineno <= line <= end_lineno
        ]
        self.branches = {
            line: value for line, value in self.branches.items()
            if lineno <= int(line) <= end_lineno
        }
        self.line_coverage = self.num_covered_lines / self.num_lines
        if self.num_branches == 0:
            self.branch_coverage = 1.0
        else:
            self.branch_coverage = self.num_covered_branches / self.num_branches

    def has_covered_line_between(self, lineno, end_lineno):
        return any(lineno <= line <= end_lineno for line in self.covered_lines)


def get_coverage_report(
    tracer: ExecutionTracer,
    suite: tsc.TestSuiteChromosome,
    metrics: set[CoverageMetric]
) -> CoverageReport:
    """Create a coverage report for the given test suite.

    Args:
        tracer: the execution tracer
        suite: The suite for which a coverage report should be generated.
        metrics: In which coverage metrics are we interested?

    Returns:
        The coverage report.
    """
    results = []
    for test_case_chromosome in suite.test_case_chromosomes:
        result = test_case_chromosome.get_last_execution_result()
        assert result is not None
        results.append(result)
    trace = ff.analyze_results(results)
    subject_properties = tracer.get_subject_properties()
    source = inspect.getsourcelines(sys.modules[Globl.module_name])[0]
    line_annotations = [
        _LineAnnotation(idx + 1, _CoverageEntry(), _CoverageEntry(), _CoverageEntry(), _CoverageEntry())
        for idx in range(len(source))
    ]

    branch_coverage = None
    branchless_code_objects = _CoverageEntry()
    branches = _CoverageEntry()
    if CoverageMetric.BRANCH in metrics:
        line_to_branchless_code_object_coverage = _get_line_to_branchless_code_object_coverage(
            subject_properties, trace
        )
        line_to_branch_coverage = _get_line_to_branch_coverage(subject_properties, trace)

        branch_coverage = ff.compute_branch_coverage(trace, subject_properties)
        for cov in line_to_branchless_code_object_coverage.values():
            branchless_code_objects += cov
        for cov in line_to_branch_coverage.values():
            branches += cov
        line_annotations = [
            line_annotation
            + _get_line_annotations_for_branch_coverage(
                idx + 1,
                line_to_branchless_code_object_coverage,
                line_to_branch_coverage,
            )
            for idx, line_annotation in enumerate(line_annotations)
        ]
    line_coverage = None
    lines = _CoverageEntry()
    if CoverageMetric.LINE in metrics:
        line_coverage = ff.compute_line_coverage(trace, subject_properties)
        covered_lines = tracer.lineids_to_linenos(trace.covered_line_ids)
        existing_lines = tracer.lineids_to_linenos(
            OrderedSet(subject_properties.existing_lines.keys())
        )
        lines += _CoverageEntry(len(covered_lines), len(existing_lines))

        def comp_line_annotation(line_no: int) -> _LineAnnotation:
            total = _CoverageEntry(
                1 if line_no in covered_lines else 0,
                1 if line_no in existing_lines else 0,
            )
            return _LineAnnotation(
                line_no,
                total=total,
                branches=_CoverageEntry(),
                branchless_code_objects=_CoverageEntry(),
                lines=total,
            )

        line_annotations = [
            line_annotation + comp_line_annotation(idx + 1)
            for idx, line_annotation in enumerate(line_annotations)
        ]

    return CoverageReport(
        module=Globl.module_name,
        source=source,
        branch_coverage=branch_coverage,
        line_coverage=line_coverage,
        branches=branches,
        branchless_code_objects=branchless_code_objects,
        lines=lines,
        line_annotations=line_annotations,
    )


def render_coverage_report(
    cov_report: CoverageReport, report_path: Path, timestamp: datetime.datetime
) -> None:
    """Render the given coverage report to the given file.

    Args:
        timestamp: When was the report created.
        cov_report: The coverage report to render
        report_path: To file where the report should be rendered to.
    """
    with report_path.open(mode="w", encoding="utf-8") as html_file:
        template = Template(
            importlib.resources.read_text("pynguin.resources", "coverage-template.html")
        )
        html_file.write(
            template.render(
                cov_report=cov_report,
                highlight=pygments.highlight,
                lexer=PythonLexer,
                formatter=HtmlFormatter,
                date=timestamp,
            )
        )


def render_xml_coverage_report(  # noqa: PLR0914
    cov_report: CoverageReport, report_path: Path, timestamp: datetime.datetime
) -> None:
    """Render the given coverage report to the given file using Cobertura XML style.

    Args:
        cov_report: The coverage report to render
        report_path: The file the report should be rendered to
        timestamp: When the report was created.
    """
    line_rate = f"{cov_report.line_coverage}"
    branch_rate = f"{cov_report.branch_coverage}"
    lines_covered = f"{cov_report.lines.covered}"
    lines_valid = f"{cov_report.lines.existing}"
    branches_covered = f"{cov_report.branches.covered + cov_report.branchless_code_objects.covered}"
    branches_valid = f"{cov_report.branches.existing + cov_report.branchless_code_objects.existing}"
    complexity = "0.0"
    version = f"pynguin-{ver.__version__}"

    report_time = f"{int(timestamp.replace(tzinfo=datetime.timezone.utc).timestamp())}"
    coverage = ET.Element(
        "coverage",
        attrib={
            "line-rate": line_rate,
            "branch-rate": branch_rate,
            "lines-covered": lines_covered,
            "lines-valid": lines_valid,
            "branches-covered": branches_covered,
            "branches-valid": branches_valid,
            "complexity": complexity,
            "version": version,
            "timestamp": report_time,
        },
    )
    sources = ET.SubElement(coverage, "sources")
    source = ET.SubElement(sources, "source")
    source.text = cov_report.module
    packages = ET.SubElement(coverage, "packages")
    package = ET.SubElement(
        packages,
        "package",
        attrib={
            "name": "",
            "line-rate": line_rate,
            "branch-rate": branch_rate,
            "complexity": complexity,
        },
    )
    classes = ET.SubElement(package, "classes")
    class_ = ET.SubElement(
        classes,
        "class",
        attrib={
            "name": "",
            "filename": cov_report.module,
            "line-rate": line_rate,
            "branch-rate": branch_rate,
            "complexity": complexity,
        },
    )
    ET.SubElement(class_, "methods")
    lines = ET.SubElement(class_, "lines")
    for line_annotation in cov_report.line_annotations:
        if line_annotation.total.existing == 0:
            continue
        attrib = {
            "number": f"{line_annotation.line_no}",
            "hits": "0",
            "branch": "false",
        }

        if line_annotation.lines.existing > 0 and line_annotation.lines.covered > 0:
            attrib["hits"] = "1"
        if (
            line_annotation.branches.existing > 0
            or line_annotation.branchless_code_objects.existing > 0
        ):
            covered = (
                line_annotation.branches.covered + line_annotation.branchless_code_objects.covered
            )
            existing = (
                line_annotation.branches.existing + line_annotation.branchless_code_objects.existing
            )
            cov = covered / existing
            cov_string = f"{cov:.0%} ({covered}/{existing})"
            attrib["condition-coverage"] = cov_string
            attrib["branch"] = "true"
            if covered > 0:
                attrib["hits"] = "1"
        ET.SubElement(lines, "line", attrib=attrib)
    tree = ET.ElementTree(coverage)
    ET.indent(tree)
    with report_path.open(mode="w", encoding="utf-8") as xml_file:
        xml_file.write('<?xml version="1.0" encoding="UTF-8"?>')
        xml_file.write(
            '<!DOCTYPE coverage SYSTEM "http://cobertura.sourceforge.net/xml/coverage-04.dtd">'
        )
        tree.write(xml_file, encoding="unicode")


def _get_line_to_branch_coverage(subject_properties, trace):
    line_to_branch_coverage = {}
    for predicate in subject_properties.existing_predicates:
        lineno = subject_properties.existing_predicates[predicate].line_no
        if lineno not in line_to_branch_coverage:
            line_to_branch_coverage[lineno] = _CoverageEntry()
        line_to_branch_coverage[lineno] += _CoverageEntry(existing=2)
        if (predicate, 0.0) in trace.true_distances.items():
            line_to_branch_coverage[lineno] += _CoverageEntry(covered=1)
        if (predicate, 0.0) in trace.false_distances.items():
            line_to_branch_coverage[lineno] += _CoverageEntry(covered=1)
    return line_to_branch_coverage


def _get_line_to_branchless_code_object_coverage(subject_properties, trace):
    line_to_branchless_code_object_coverage = {}
    for code in subject_properties.branch_less_code_objects:
        lineno = subject_properties.existing_code_objects[code].code_object.co_firstlineno
        if lineno not in line_to_branchless_code_object_coverage:
            line_to_branchless_code_object_coverage[lineno] = _CoverageEntry()
        line_to_branchless_code_object_coverage[lineno] += _CoverageEntry(existing=1)
        if code in trace.executed_code_objects:
            line_to_branchless_code_object_coverage[lineno] += _CoverageEntry(covered=1)
    return line_to_branchless_code_object_coverage


def _get_line_annotations_for_branch_coverage(
    lineno: int,
    code_object_coverage: dict[int, _CoverageEntry],
    predicate_coverage: dict[int, _CoverageEntry],
) -> _LineAnnotation:
    """Compute line annotation for branch coverage of the given line no.

    Args:
        lineno: The lineno for which we should generate the information.
        code_object_coverage: code object coverage data
        predicate_coverage: predicate coverage data

    Returns:
        LineAnnotation data for the given line.
    """
    total = _CoverageEntry()
    branches = _CoverageEntry()
    branchless_code_objects = _CoverageEntry()
    if lineno in code_object_coverage:
        branchless_code_objects = code_object_coverage[lineno]
        total += branchless_code_objects
    if lineno in predicate_coverage:
        branches = predicate_coverage[lineno]
        total += branches
    return _LineAnnotation(lineno, total, branches, branchless_code_objects, _CoverageEntry())
