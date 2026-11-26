"""Data provenance tracking and taxonomy for GWLab artifacts."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ProvenanceCategory(str, Enum):
    """Authoritative scientific data classification categories."""
    SAVED_POSTERIOR = "SAVED POSTERIOR"
    DERIVED_STATISTIC = "DERIVED STATISTIC"
    ILLUSTRATIVE_MOCK = "ILLUSTRATIVE MOCK"


class ArtifactValidationStatus(str, Enum):
    """Integrity and schema validation state for scientific artifacts."""
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ProvenanceRecord:
    """Immutable provenance record tracking origin and validation of scientific data."""
    artifact_path: str
    run_id: str
    model: str
    seed: Optional[int]
    category: ProvenanceCategory
    parameter_order: Optional[List[str]] = None
    parameter_units: Optional[Dict[str, str]] = None
    validation_status: ArtifactValidationStatus = ArtifactValidationStatus.VALID
    validation_notes: List[str] = field(default_factory=list)

    def summary_badge_text(self) -> str:
        """Formatted badge text for workstation display."""
        seed_part = f" • Seed {self.seed}" if self.seed is not None else ""
        return f"[{self.category.value}] {self.run_id} ({self.model}{seed_part})"
