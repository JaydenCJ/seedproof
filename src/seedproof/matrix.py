"""Run matrix: group N recorded runs into equivalence classes by output.

Two runs land in the same class when their token streams are identical on
the chosen basis. The matrix then asks the question users actually argue
about: *which configuration axis explains the split?* For every config field
that varies across the runs, the axis analysis reports whether that field's
values partition the runs exactly like the output classes do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .align import first_divergence
from .errors import MatrixError
from .record import RunRecord

#: Axis relations, strongest first.
RELATION_EXPLAINS = "explains"      # field partition == class partition
RELATION_CORRELATES = "correlates"  # same value => same class, but coarser
RELATION_MIXED = "mixed"            # same value lands in different classes


@dataclass
class RunClass:
    """One equivalence class: every member produced this exact stream."""

    label: str
    fingerprint: str
    records: List[RunRecord]

    @property
    def members(self) -> List[str]:
        return [record.name for record in self.records]

    @property
    def representative(self) -> RunRecord:
        return self.records[0]


@dataclass
class AxisFinding:
    """How one varying config field relates to the class split."""

    field: str
    relation: str
    #: value (rendered as JSON) -> sorted class labels it appears in
    value_classes: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class PairDivergence:
    """First divergence between the representatives of two classes."""

    label_a: str
    label_b: str
    kind: str
    index: Optional[int]
    a_text: Optional[str]
    b_text: Optional[str]


@dataclass
class MatrixReport:
    basis: str
    prompt_sha256: str
    records: List[RunRecord]
    classes: List[RunClass]
    axes: List[AxisFinding]
    combined_axes: Optional[Tuple[str, ...]]
    pairwise: List[PairDivergence]

    @property
    def consistent(self) -> bool:
        """True when every run produced the same token stream."""
        return len(self.classes) == 1


def _config_value(record: RunRecord, field_name: str) -> Any:
    if field_name.startswith("extra."):
        return record.config.extra.get(field_name[len("extra."):])
    return getattr(record.config, field_name)


def _field_names(records: Sequence[RunRecord]) -> List[str]:
    names = ["model", "backend", "device", "quant", "seed",
             "sampler", "temperature", "top_k", "top_p"]
    extra_keys: List[str] = []
    for record in records:
        for key in record.config.extra:
            if key not in extra_keys:
                extra_keys.append(key)
    return names + [f"extra.{key}" for key in sorted(extra_keys)]


def _class_labels(count: int) -> List[str]:
    """A, B, ... Z, AA, AB, ... — enough labels for any realistic matrix."""
    labels = []
    for i in range(count):
        label = ""
        n = i
        while True:
            label = chr(ord("A") + n % 26) + label
            n = n // 26 - 1
            if n < 0:
                break
        labels.append(label)
    return labels


def build_matrix(records: Sequence[RunRecord], basis: str = "auto") -> MatrixReport:
    """Group runs into output classes and analyze which config axes split them.

    All runs must share one prompt — comparing generations of different
    prompts is meaningless, so that is a hard error, not a warning.
    """
    if len(records) < 2:
        raise MatrixError(f"need at least 2 records to build a matrix, got {len(records)}")
    prompts = {record.prompt_sha256() for record in records}
    if len(prompts) > 1:
        detail = ", ".join(
            f"{record.name}={record.prompt_sha256()[:12]}" for record in records
        )
        raise MatrixError(
            f"records answer {len(prompts)} different prompts ({detail}); "
            "a matrix only makes sense over one prompt"
        )

    if basis == "auto":
        resolved = "id" if all(r.ids() is not None for r in records) else "text"
    elif basis in ("id", "text"):
        resolved = basis
    else:
        raise MatrixError(f"unknown basis '{basis}' (use auto, id, or text)")

    # Group by stream fingerprint, preserving first-seen order.
    by_fingerprint: Dict[str, List[RunRecord]] = {}
    for record in records:
        by_fingerprint.setdefault(record.stream_fingerprint(resolved), []).append(record)
    # Largest class first; sorted() is stable, so equal-sized classes keep
    # their first-seen order.
    ordered = sorted(by_fingerprint.items(), key=lambda item: -len(item[1]))
    labels = _class_labels(len(ordered))
    classes = [
        RunClass(label=labels[i], fingerprint=fp, records=members)
        for i, (fp, members) in enumerate(ordered)
    ]
    class_of = {
        record.name: cls.label for cls in classes for record in cls.records
    }

    axes = _analyze_axes(records, classes, class_of)
    combined = _find_combined_axes(records, classes, class_of, axes)
    pairwise = _pairwise(classes, resolved)

    return MatrixReport(
        basis=resolved,
        prompt_sha256=next(iter(prompts)),
        records=list(records),
        classes=classes,
        axes=axes,
        combined_axes=combined,
        pairwise=pairwise,
    )


def _analyze_axes(
    records: Sequence[RunRecord],
    classes: List[RunClass],
    class_of: Dict[str, str],
) -> List[AxisFinding]:
    findings: List[AxisFinding] = []
    for field_name in _field_names(records):
        value_classes: Dict[str, set] = {}
        for record in records:
            rendered = json.dumps(_config_value(record, field_name), sort_keys=True)
            value_classes.setdefault(rendered, set()).add(class_of[record.name])
        if len(value_classes) < 2:
            continue  # constant across the matrix: not an axis
        mixed = any(len(cls_set) > 1 for cls_set in value_classes.values())
        if mixed:
            relation = RELATION_MIXED
        elif len(value_classes) == len(classes):
            # value -> class is a function and counts match => bijection.
            relation = RELATION_EXPLAINS
        else:
            relation = RELATION_CORRELATES
        findings.append(
            AxisFinding(
                field=field_name,
                relation=relation,
                value_classes={
                    value: sorted(cls_set) for value, cls_set in value_classes.items()
                },
            )
        )
    return findings


def _find_combined_axes(
    records: Sequence[RunRecord],
    classes: List[RunClass],
    class_of: Dict[str, str],
    axes: List[AxisFinding],
) -> Optional[Tuple[str, ...]]:
    """When no single field explains the split, try pairs of varying fields.

    A (backend, quant) matrix commonly splits three ways where neither field
    alone is a function of the class — but the pair is. Reporting that pair
    turns "nothing explains it" into an actionable answer.
    """
    if len(classes) < 2 or any(a.relation == RELATION_EXPLAINS for a in axes):
        return None
    varying = [a.field for a in axes]
    for pair in combinations(varying, 2):
        value_classes: Dict[Tuple[str, str], set] = {}
        for record in records:
            key = tuple(
                json.dumps(_config_value(record, f), sort_keys=True) for f in pair
            )
            value_classes.setdefault(key, set()).add(class_of[record.name])
        is_function = all(len(cls_set) == 1 for cls_set in value_classes.values())
        classes_hit = set()
        for cls_set in value_classes.values():
            classes_hit |= cls_set
        if is_function and len(classes_hit) == len(classes):
            return pair
    return None


def _pairwise(classes: List[RunClass], basis: str) -> List[PairDivergence]:
    results: List[PairDivergence] = []
    for cls_a, cls_b in combinations(classes, 2):
        divergence = first_divergence(cls_a.representative, cls_b.representative, basis)
        results.append(
            PairDivergence(
                label_a=cls_a.label,
                label_b=cls_b.label,
                kind=divergence.kind,
                index=divergence.index,
                a_text=divergence.a_token.text if divergence.a_token else None,
                b_text=divergence.b_token.text if divergence.b_token else None,
            )
        )
    return results
