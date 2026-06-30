"""Murphy-Hill taxonomy mappings for refactoring operations."""

from __future__ import annotations

from typing import Any, Optional

REFACTORING_TAXONOMY_VERSION = "v1"

MURPHY_HILL_LOW_LEVEL_TYPES = {
    # RefactoringMiner-style concrete operations
    "Add Parameter Modifier",
    "Add Variable Annotation",
    "Add Variable Modifier",
    "Assert Throws",
    "Assert Timeout",
    "Change Variable Type",
    "Extract Variable",
    "Inline Variable",
    "Invert Condition",
    "Merge Catch",
    "Merge Conditional",
    "Merge Variable",
    "Modify Variable Annotation",
    "Move Code",
    "Remove Parameter Modifier",
    "Remove Variable Annotation",
    "Remove Variable Modifier",
    "Rename Parameter",
    "Rename Variable",
    "Replace Anonymous with Lambda",
    "Replace Attribute",
    "Replace Conditional With Ternary",
    "Replace Conditional with Assumption",
    "Replace Generic With Diamond",
    "Replace Loop with Pipeline",
    "Replace Pipeline with Loop",
    "Split Conditional",
    "Split Variable",
    "Try With Resources",
}

MURPHY_HILL_MEDIUM_LEVEL_TYPES = {
    # RefactoringMiner-style concrete operations
    "Change Attribute Type",
    "Change Parameter Type",
    "Change Return Type",
    "Extract And Move Method",
    "Extract Attribute",
    "Extract Class",
    "Extract Fixture",
    "Extract Method",
    "Extract Subclass",
    "Inline Attribute",
    "Inline Method",
    "Localize Parameter",
    "Merge Method",
    "Move And Inline Method",
    "Parameterize Attribute",
    "Replace Anonymous with Class",
    "Replace Attribute with Variable",
    "Replace Variable with Attribute",
    "Split Class",
    "Split Method",

    # RefDiff relationship labels
    "Extract",
    "Extract And Move",
    "Inline",
}

MURPHY_HILL_HIGH_LEVEL_TYPES = {
    # RefactoringMiner-style concrete operations
    "Add Attribute Annotation",
    "Add Attribute Modifier",
    "Add Class Annotation",
    "Add Class Modifier",
    "Add Method Annotation",
    "Add Method Modifier",
    "Add Parameter",
    "Add Parameter Annotation",
    "Add Thrown Exception Type",
    "Change Attribute Access Modifier",
    "Change Class Access Modifier",
    "Change Method Access Modifier",
    "Change Signature of Method",
    "Change Thrown Exception Type",
    "Change Type Declaration Kind",
    "Collapse Hierarchy",
    "Encapsulate Attribute",
    "Extract Interface",
    "Extract Superclass",
    "Extract Supertype",
    "Merge Attribute",
    "Merge Class",
    "Merge Package",
    "Merge Parameter",
    "Modify Attribute Annotation",
    "Modify Class Annotation",
    "Modify Method Annotation",
    "Modify Parameter Annotation",
    "Move And Rename Attribute",
    "Move And Rename Class",
    "Move And Rename Method",
    "Move Annotation",
    "Move Attribute",
    "Move Class",
    "Move Method",
    "Move Package",
    "Parameterize Test",
    "Parameterize Variable",
    "Pull Up Attribute",
    "Pull Up Method",
    "Push Down Attribute",
    "Push Down Method",
    "Remove Attribute Annotation",
    "Remove Attribute Modifier",
    "Remove Class Annotation",
    "Remove Class Modifier",
    "Remove Method Annotation",
    "Remove Method Modifier",
    "Remove Parameter",
    "Remove Parameter Annotation",
    "Remove Thrown Exception Type",
    "Rename Attribute",
    "Rename Class",
    "Rename Method",
    "Rename Package",
    "Reorder Parameter",
    "Split Attribute",
    "Split Package",
    "Split Parameter",

    # RefDiff relationship labels
    "Change Signature",
    "Change Signature of Function",
    "Convert Type",
    "Internal Move",
    "Internal Move And Rename",
    "Move",
    "Move And Rename",
    "Pull Up",
    "Push Down",
    "Rename",
}


def classify_murphy_hill_level(refactoring_type: Optional[str]) -> Optional[str]:
    """Return the Murphy-Hill abstraction level for a canonical refactoring type."""
    if not refactoring_type:
        return None
    if refactoring_type in MURPHY_HILL_LOW_LEVEL_TYPES:
        return "low"
    if refactoring_type in MURPHY_HILL_MEDIUM_LEVEL_TYPES:
        return "medium"
    if refactoring_type in MURPHY_HILL_HIGH_LEVEL_TYPES:
        return "high"
    return None


def classify_refactoring_taxonomy(refactoring_type: Optional[str]) -> dict[str, Any]:
    """Return taxonomy metadata for one canonical refactoring type."""
    if not refactoring_type:
        return {
            "murphy_hill_level": None,
            "_meta": {"version": REFACTORING_TAXONOMY_VERSION, "source": "empty"},
        }
    refactoring_type = str(refactoring_type).strip()
    if refactoring_type in MURPHY_HILL_LOW_LEVEL_TYPES:
        return {
            "murphy_hill_level": "low",
            "_meta": {"version": REFACTORING_TAXONOMY_VERSION, "sources": {"murphy_hill_level": "exact"}},
        }
    if refactoring_type in MURPHY_HILL_MEDIUM_LEVEL_TYPES:
        return {
            "murphy_hill_level": "medium",
            "_meta": {"version": REFACTORING_TAXONOMY_VERSION, "sources": {"murphy_hill_level": "exact"}},
        }
    if refactoring_type in MURPHY_HILL_HIGH_LEVEL_TYPES:
        return {
            "murphy_hill_level": "high",
            "_meta": {"version": REFACTORING_TAXONOMY_VERSION, "sources": {"murphy_hill_level": "exact"}},
        }
    return {
        "murphy_hill_level": None,
        "_meta": {"version": REFACTORING_TAXONOMY_VERSION, "sources": {"murphy_hill_level": "unmapped"}},
    }
