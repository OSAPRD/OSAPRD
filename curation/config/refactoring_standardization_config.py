"""Analysis-side refactoring standardization configuration."""

from __future__ import annotations

import re

# Explicit per-tool mappings used by analysis-side restandardization.
# The right-hand side is the canonical label used downstream in analysis.

# RefDiff may emit relationship edges that align unchanged entities. They are
# parser metadata, not refactoring operations, so they are skipped before
# taxonomy classification instead of being canonicalized.
NON_REFACTORING_REFDIFF_TYPES: set[str] = {"Same"}

REFDIFF_REFACTORING_TYPE_MAP: dict[str, str] = {
    "Convert Type": "Convert Type",
    "Change Signature of Method/Function": "Change Signature of Method",
    "Pull Up Method": "Pull Up Method",
    "Push Down Method": "Push Down Method",
    "Rename": "Rename",
    "Move": "Move",
    "Move and Rename": "Move And Rename",
    "Move And Rename": "Move And Rename",
    "Extract Supertype (e.g., Class/Interface)": "Extract Supertype",
    "Extract Method/Function": "Extract Method",
    "Inline Method/Function": "Inline Method",
}


REFACTORINGMINER_REFACTORING_TYPE_MAP: dict[str, str] = {
    # From Fowler's book
    "Extract Method": "Extract Method",
    "Inline Method": "Inline Method",
    "Rename Method": "Rename Method",
    "Move Method": "Move Method",
    "Move Attribute": "Move Attribute",
    "Pull Up Method": "Pull Up Method",
    "Pull Up Attribute": "Pull Up Attribute",
    "Push Down Method": "Push Down Method",
    "Push Down Attribute": "Push Down Attribute",
    "Extract Superclass": "Extract Superclass",
    "Extract Interface": "Extract Interface",
    "Move Class": "Move Class",
    "Rename Class": "Rename Class",
    "Extract and Move Method": "Extract And Move Method",
    "Extract And Move Method": "Extract And Move Method",
    "Rename Package": "Rename Package",
    "Move and Rename Class": "Move And Rename Class",
    "Move And Rename Class": "Move And Rename Class",
    "Extract Class": "Extract Class",
    "Extract Subclass": "Extract Subclass",
    "Extract Variable": "Extract Variable",
    "Inline Variable": "Inline Variable",
    "Parameterize Variable": "Parameterize Variable",
    "Extract Attribute": "Extract Attribute",
    "Move and Rename Method": "Move And Rename Method",
    "Move And Rename Method": "Move And Rename Method",
    "Move and Inline Method": "Move And Inline Method",
    "Move And Inline Method": "Move And Inline Method",
    "Encapsulate Attribute": "Encapsulate Attribute",
    "Parameterize Attribute": "Parameterize Attribute",
    "Move Package": "Move Package",
    "Split Package": "Split Package",
    "Merge Package": "Merge Package",
    "Localize Parameter": "Localize Parameter",
    "Collapse Hierarchy": "Collapse Hierarchy",
    "Merge Class": "Merge Class",
    "Inline Attribute": "Inline Attribute",
    "Split Class": "Split Class",
    "Split Conditional": "Split Conditional",
    "Invert Condition": "Invert Condition",
    "Merge Conditional": "Merge Conditional",
    "Merge Method": "Merge Method",
    "Split Method": "Split Method",
    "Move Code (between methods)": "Move Code",
    "Move Code": "Move Code",
    # API changes
    "Rename Variable": "Rename Variable",
    "Rename Parameter": "Rename Parameter",
    "Rename Attribute": "Rename Attribute",
    "Move and Rename Attribute": "Move And Rename Attribute",
    "Move And Rename Attribute": "Move And Rename Attribute",
    "Replace Variable with Attribute": "Replace Variable with Attribute",
    "Replace Attribute (with Attribute)": "Replace Attribute",
    "Replace Attribute": "Replace Attribute",
    "Merge Variable": "Merge Variable",
    "Merge Parameter": "Merge Parameter",
    "Merge Attribute": "Merge Attribute",
    "Split Variable": "Split Variable",
    "Split Parameter": "Split Parameter",
    "Split Attribute": "Split Attribute",
    "Change Variable Type": "Change Variable Type",
    "Change Parameter Type": "Change Parameter Type",
    "Change Return Type": "Change Return Type",
    "Change Attribute Type": "Change Attribute Type",
    "Add Method Annotation": "Add Method Annotation",
    "Remove Method Annotation": "Remove Method Annotation",
    "Modify Method Annotation": "Modify Method Annotation",
    "Add Attribute Annotation": "Add Attribute Annotation",
    "Remove Attribute Annotation": "Remove Attribute Annotation",
    "Modify Attribute Annotation": "Modify Attribute Annotation",
    "Add Class Annotation": "Add Class Annotation",
    "Remove Class Annotation": "Remove Class Annotation",
    "Modify Class Annotation": "Modify Class Annotation",
    "Add Parameter Annotation": "Add Parameter Annotation",
    "Remove Parameter Annotation": "Remove Parameter Annotation",
    "Modify Parameter Annotation": "Modify Parameter Annotation",
    "Add Variable Annotation": "Add Variable Annotation",
    "Remove Variable Annotation": "Remove Variable Annotation",
    "Modify Variable Annotation": "Modify Variable Annotation",
    "Add Parameter": "Add Parameter",
    "Remove Parameter": "Remove Parameter",
    "Reorder Parameter": "Reorder Parameter",
    "Add Thrown Exception Type": "Add Thrown Exception Type",
    "Remove Thrown Exception Type": "Remove Thrown Exception Type",
    "Change Thrown Exception Type": "Change Thrown Exception Type",
    "Change Method Access Modifier": "Change Method Access Modifier",
    "Change Attribute Access Modifier": "Change Attribute Access Modifier",
    "Replace Attribute with Variable": "Replace Attribute with Variable",
    "Add Method Modifier (final, static, abstract, synchronized)": "Add Method Modifier",
    "Remove Method Modifier (final, static, abstract, synchronized)": "Remove Method Modifier",
    "Add Attribute Modifier (final, static, transient, volatile)": "Add Attribute Modifier",
    "Remove Attribute Modifier (final, static, transient, volatile)": "Remove Attribute Modifier",
    "Add Variable Modifier (final)": "Add Variable Modifier",
    "Add Parameter Modifier (final)": "Add Parameter Modifier",
    "Remove Variable Modifier (final)": "Remove Variable Modifier",
    "Remove Parameter Modifier (final)": "Remove Parameter Modifier",
    "Change Class Access Modifier": "Change Class Access Modifier",
    "Add Class Modifier (final, static, abstract)": "Add Class Modifier",
    "Remove Class Modifier (final, static, abstract)": "Remove Class Modifier",
    "Change Type Declaration Kind (class, interface, enum, annotation, record)": "Change Type Declaration Kind",
    "Move Annotation": "Move Annotation",
    # Migrations
    "Replace Loop with Pipeline": "Replace Loop with Pipeline",
    "Replace Anonymous with Lambda": "Replace Anonymous with Lambda",
    "Replace Pipeline with Loop": "Replace Pipeline with Loop",
    "Merge Catch": "Merge Catch",
    "Replace Anonymous with Class": "Replace Anonymous with Class",
    "Replace Generic With Diamond": "Replace Generic With Diamond",
    "Try With Resources": "Try With Resources",
    "Replace Conditional With Ternary": "Replace Conditional With Ternary",
    # Test-specific
    "Parameterize Test (JUnit 5 @ParameterizedTest with @ValueSource)": "Parameterize Test",
    "Parameterize Test": "Parameterize Test",
    "Assert Throws": "Assert Throws",
    "Assert Timeout": "Assert Timeout",
    "Replace Conditional with Assumption": "Replace Conditional with Assumption",
    "Extract Fixture": "Extract Fixture",
}


# Flattened alias export used by the live analysis restandardization code.
REFACTORING_TYPE_ALIASES: dict[str, str] = {
    **REFDIFF_REFACTORING_TYPE_MAP,
    **REFACTORINGMINER_REFACTORING_TYPE_MAP,
    # Additional observed syntactic variants.
    "EXTRACT": "Extract",
    "INTERNAL_MOVE": "Internal Move",
    "MOVE_AND_RENAME": "Move And Rename",
    "Move Rename": "Move And Rename",
    "Move_Rename": "Move And Rename",
    "Move Rename Method": "Move And Rename Method",
    "Move Rename Class": "Move And Rename Class",
    "Move Rename Attribute": "Move And Rename Attribute",
    "Extract Move": "Extract And Move Method",
    "Extract_Move": "Extract And Move Method",
    "Extract And Move": "Extract And Move Method",
}


def _canonical_refactoring_lookup_key(value: str | None) -> str:
    """Normalize a raw refactoring label into a lookup key."""
    text = str(value or "").strip().replace("&", " and ")
    text = re.sub(r"[_/\-]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return " ".join(text.lower().split())


NORMALIZED_REFACTORING_TYPE_ALIASES: dict[str, str] = {
    _canonical_refactoring_lookup_key(raw_label): standardized_label
    for raw_label, standardized_label in REFACTORING_TYPE_ALIASES.items()
    if _canonical_refactoring_lookup_key(raw_label)
}


# Raw labels that need additional context from description/raw payload in order
# to become a more specific standardized operation type.
GENERIC_REFACTORING_TYPES: set[str] = {
    "Move",
    "Rename",
    "Inline",
    "Extract",
    "Move And Rename",
    "Extract And Move Method",
    "Internal Move",
    "Move Code",
}


NORMALIZED_CANONICAL_REFACTORING_LABELS: dict[str, str] = {
    _canonical_refactoring_lookup_key(label): label
    for label in sorted(set(REFACTORING_TYPE_ALIASES.values()) | GENERIC_REFACTORING_TYPES)
    if _canonical_refactoring_lookup_key(label)
}

NORMALIZED_NON_REFACTORING_REFDIFF_TYPES: set[str] = {
    _canonical_refactoring_lookup_key(label)
    for label in NON_REFACTORING_REFDIFF_TYPES
    if _canonical_refactoring_lookup_key(label)
}


def canonicalize_refactoring_type(raw_type: str | None) -> str | None:
    """Map a raw tool-specific refactoring label to the canonical label."""
    if raw_type is None:
        return None
    normalized = " ".join(str(raw_type).strip().replace("_", " ").split())
    normalized_key = _canonical_refactoring_lookup_key(normalized)
    if not normalized_key:
        return None
    if normalized_key in NORMALIZED_REFACTORING_TYPE_ALIASES:
        return NORMALIZED_REFACTORING_TYPE_ALIASES[normalized_key]
    return NORMALIZED_CANONICAL_REFACTORING_LABELS.get(normalized_key)


def is_non_refactoring_relationship_type(raw_type: str | None) -> bool:
    """Return True when a raw tool label is known not to be a refactoring."""
    normalized_key = _canonical_refactoring_lookup_key(raw_type)
    return normalized_key in NORMALIZED_NON_REFACTORING_REFDIFF_TYPES
