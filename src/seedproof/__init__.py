"""seedproof — token-stream divergence forensics for recorded generations.

Compare recorded runs across seeds, backends, and quantizations; pinpoint
the first divergent token; and classify the cause with log-probability
evidence. Pure standard library, fully offline.

Public API:

- records: :class:`RunRecord`, :class:`RunConfig`, :class:`Token`,
  :class:`TokenChoice`, :func:`load`, :func:`loads`, :func:`save`,
  :func:`dumps`, :func:`compare_configs`
- ingest:  :func:`ingest` (formats: generic, jsonl, sse)
- align:   :func:`first_divergence`, :func:`prefix_drift`,
  :func:`analyze_tie`, :func:`find_resync`
- verdict: :func:`diagnose` -> :class:`Diagnosis`
- matrix:  :func:`build_matrix` -> :class:`MatrixReport`
"""

from .align import (
    DEFAULT_TIE_EPSILON,
    Divergence,
    PrefixDrift,
    Resync,
    TieAnalysis,
    analyze_tie,
    choose_basis,
    find_resync,
    first_divergence,
    prefix_drift,
)
from .errors import (
    AlignError,
    IngestError,
    MatrixError,
    RecordError,
    SeedproofError,
)
from .forensics import VERDICTS, Diagnosis, Evidence, diagnose
from .ingest import FORMATS, ingest
from .matrix import AxisFinding, MatrixReport, PairDivergence, RunClass, build_matrix
from .record import (
    RECORD_VERSION,
    RunConfig,
    RunRecord,
    Token,
    TokenChoice,
    compare_configs,
    dumps,
    load,
    loads,
    save,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AlignError",
    "AxisFinding",
    "DEFAULT_TIE_EPSILON",
    "Diagnosis",
    "Divergence",
    "Evidence",
    "FORMATS",
    "IngestError",
    "MatrixError",
    "MatrixReport",
    "PairDivergence",
    "PrefixDrift",
    "RECORD_VERSION",
    "RecordError",
    "Resync",
    "RunClass",
    "RunConfig",
    "RunRecord",
    "SeedproofError",
    "TieAnalysis",
    "Token",
    "TokenChoice",
    "VERDICTS",
    "analyze_tie",
    "build_matrix",
    "choose_basis",
    "compare_configs",
    "diagnose",
    "dumps",
    "find_resync",
    "first_divergence",
    "ingest",
    "load",
    "loads",
    "prefix_drift",
    "save",
]
