"""Unified taxonomy mappings and heuristics for standardized code smells."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

from curation.config.code_smell_standardization_config import CODE_SMELL_RULE_MAP

CODE_SMELL_TAXONOMY_VERSION = "v2"


def _normalized(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _normalized_list(values: Optional[Iterable[Any]]) -> list[str]:
    return [_normalized(str(value)) for value in (values or []) if _normalized(str(value))]


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


MANTYLA_RULE_MAP: dict[str, str] = {}

_BLOATERS = {
    "Long Method",
    "Complex Method",
    "Long Parameter List",
    "Data Clumps / Too Many Parameters",
    "Long Identifier",
    "Long Statement",
    "Long Lambda Function",
    "Overly Long File",
    "Too Many Responsibilities",
    "God Component",
    "Multifaceted Abstraction",
    "Insufficient Modularization",
    "Feature Concentration",
    "Dense Structure",
    "Deep Nesting",
    "Callback Hell",
    "Excessive Nesting",
    "Variable Scope Too Large",
    "Primitive Obsession",
    "Magic Number",
    "Low-Level Obsession",
    "Overcomplicated Control Flow",
    "Overcomplicated Looping",
    "Loop Logic Smell",
    "Verbose Conditional",
    "Verbose Logic",
    "Verbose Type Declaration",
    "Verbose Construction",
    "Escaping Complexity",
    "Temporary Object",
    "Excessive Copying",
    "Inefficient Data Handling",
    "Pass-by-Value Smell",
    "Inefficient Data Passing",
    "Inefficient Abstraction",
    "Inefficient Loop",
    "Inefficient Search",
    "Inefficient Construction",
    "Performance Bottleneck",
    "Performance Smell",
    "Resource Waste",
    "IO Performance Smell",
    "Minor Inefficiency",
    "Loop Inefficiency",
    "Oversized Representation",
    "Ineffective Move Semantics",
    "Missed Move Optimization",
    "Unnecessary Iteration",
}

_OO_ABUSERS = {
    "Switch Statements Too Large",
    "Large Switch Statement",
    "Missing Default",
    "Temporary Field",
    "Temporary Field / Unused State",
    "Rebellious Hierarchy",
    "Broken Hierarchy",
    "Wide Hierarchy",
    "Deep Hierarchy",
    "Missing Hierarchy",
    "Multipath Hierarchy",
    "Cyclic Hierarchy",
    "Refused Bequest",
    "Fragile Inheritance",
    "Inheritance Misuse",
    "Incomplete Object Semantics",
    "Imperative Abstraction",
    "Abstract Function Call From Constructor",
    "Constructor Initialization",
    "Incorrect Initialization",
    "Poor Construction",
    "Incomplete Initialization",
    "Redundant Initialization",
    "Primitive Misuse",
    "Type Blind Conversion",
    "Unsafe Type Conversion",
    "Unsafe Conversion",
}

_CHANGE_PREVENTERS = {
    "Complex Conditional",
    "Conditional Complexity",
    "Complex Conditional / Boolean Complexity",
    "Spaghetti Code",
    "Unnecessary Complexity",
    "Incomplete Conditional Logic",
    "Repeated Conditions",
    "Confusing Conditional",
    "Boolean Blindness / Confusing Booleans",
    "Suspicious Control Flow",
    "Faulty Control Flow",
    "Ambiguous Control Flow",
    "Obscure Control Flow",
    "One-Iteration Loops",
    "Inconsistent Behavior",
    "Inconsistent State",
    "Mutable Shared State",
    "Mutable State Smell",
    "Global Initialization Dependency",
    "Unstable Dependency",
    "Broken Modularization",
    "Scattered Functionality",
    "Cyclic Dependency",
    "Cyclically-Dependent Modularization",
    "Resource Leak",
    "Memory Leak",
    "Manual Resource Management",
    "Unsafe Resource Handling",
    "Inconsistent Resource Management",
    "Unsafe Memory Management",
    "Unsafe Ownership",
    "Ownership Mismanagement",
    "Ownership Confusion",
    "Lifetime Mismanagement",
    "Escaping Temporary",
    "Dangling Reference",
    "Invalid Object State",
    "Unsafe State Handling",
    "Unsafe Collection Handling",
    "Collection Misuse",
    "Incorrect Collection Usage",
    "Poor Exception Handling",
    "Incomplete Error Handling",
    "Exception Handling",
    "Exception Swallowing",
    "Suspicious Equality / Hidden Bugs",
    "Hidden Behavior",
    "Hidden Semantics",
    "Hidden Side Effects",
    "Ambiguous Merge Key",
    "Chain Indexing",
    "Forward Bypass",
}

_DISPENSABLES = {
    "Duplicate Code",
    "Duplicate Logic",
    "Duplicate Branches",
    "Duplicate Abstraction",
    "Redundant Code",
    "Redundant Logic",
    "Redundant Control Flow",
    "Redundant Conditional Logic",
    "Redundant Statements",
    "Redundant State",
    "Redundant Copying",
    "Lazy Class / Empty Abstraction",
    "Lazy / Redundant Statements",
    "Boilerplate Code",
    "Data Class",
    "Dead Code",
    "Dead Code / Lazy Class",
    "Dead Store / Dead Code",
    "Dead State",
    "Dead Parameter",
    "Dead Abstraction",
    "Dead Branches",
    "Write-only Variable",
    "Temporary Variable",
    "Unused Collections Or Values",
    "Unnecessary Abstraction",
    "Unutilized Abstraction",
    "Ignored Behavior",
    "Ignored Test",
    "Empty Catch Clause",
    "Empty Test",
    "Lazy Class",
    "Speculative Generality",
    "Obsolete API Usage",
    "Obsolete Abstraction",
    "Obsolete Language Usage",
    "Obsolete Typedef Style",
    "Obsolete Ownership Model",
    "Preprocessor Abuse",
    "Macro Abuse",
}

_ENCAPSULATORS = {
    "Deficient Encapsulation",
    "Unexploited Encapsulation",
    "Global Data",
    "Global State",
    "Global State Abuse",
    "Poor Self-Documentation",
}

_COUPLERS = {
    "Feature Envy",
    "Long Message Chain",
    "Law of Demeter Violation",
    "Hard-wired Dependencies",
    "Excessive Dependency",
    "Hub-like Modularization",
    "Excessive Coupling / Recursive Complexity",
    "Ambiguous API Design",
    "Confusing API Usage",
    "Confusing API Calls",
    "Unsafe Dynamic Code",
    "Inappropriate Intimacy with Container",
    "Temporal Coupling",
}

_OTHERS = {
    "Ambiguous Interface",
    "Broken NaN Check",
    "Parsing Error",
    "Poor Naming",
    "Confusing Names",
    "Obscure Intent",
    "Obscure Type Semantics",
    "Incorrect Memory Semantics",
    "Inconsistent Style",
    "Incorrect String Logic",
    "Assertion Roulette",
    "Missing Assertion",
    "Conditional Test Logic",
    "Eager Test",
    "Unknown Test",
    "Fragile Assignment Semantics",
    "Unsafe Memory Operations",
    "Unsafe String Logic",
    "Unsafe Exception Flow",
    "Use After Free",
    "Undefined State",
}


def _build_primary_mantyla_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for label in _BLOATERS:
        mapping[label.lower()] = "bloaters"
    for label in _OO_ABUSERS:
        mapping[label.lower()] = "object_orientation_abusers"
    for label in _CHANGE_PREVENTERS:
        mapping[label.lower()] = "change_preventers"
    for label in _DISPENSABLES:
        mapping[label.lower()] = "dispensables"
    for label in _ENCAPSULATORS:
        mapping[label.lower()] = "encapsulators"
    for label in _COUPLERS:
        mapping[label.lower()] = "couplers"
    for label in _OTHERS:
        mapping[label.lower()] = "others"
    return mapping


PRIMARY_MANTYLA_RULE_MAP: dict[str, str] = _build_primary_mantyla_map()
MANTYLA_RULE_MAP.update(PRIMARY_MANTYLA_RULE_MAP)


def _from_maps(
    smell_type: Optional[str],
    *,
    rule_map: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    normalized_smell = _normalized(smell_type)
    if normalized_smell and normalized_smell in rule_map:
        return rule_map[normalized_smell], "exact_smell_type"
    return None, None


def _heuristic_text(
    *,
    rule_id: Optional[str],
    category: Optional[str],
    message: Optional[str],
    tags: Optional[Iterable[Any]],
    language: Optional[str],
    file_path: Optional[str],
) -> str:
    parts = [
        _normalized(rule_id),
        _normalized(category),
        _normalized(message),
        _normalized(language),
        _normalized(file_path),
        " ".join(_normalized_list(tags)),
    ]
    return " ".join(part for part in parts if part)

def _bootstrap_standardized_smell_rule_maps() -> None:
    standardized_smells = sorted(
        {
            _normalized(str(label))
            for label in CODE_SMELL_RULE_MAP.values()
            if _normalized(str(label))
        }
    )
    for smell in standardized_smells:
        if smell not in MANTYLA_RULE_MAP:
            MANTYLA_RULE_MAP[smell] = PRIMARY_MANTYLA_RULE_MAP.get(smell, "others")


_bootstrap_standardized_smell_rule_maps()


def _taxonomy_value_with_source(
    *,
    rule_id: Optional[str],
    rule_map: Dict[str, str],
    default: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    mapped, source = _from_maps(
        rule_id,
        rule_map=rule_map,
    )
    if mapped:
        return mapped, source or "exact_smell_type"
    return default, "default"


def classify_code_smell_taxonomy(
    *,
    rule_id: Optional[str],
    category: Optional[str],
    message: Optional[str] = None,
    tags: Optional[Iterable[Any]] = None,
    language: Optional[str] = None,
    file_path: Optional[str] = None,
) -> Dict[str, Any]:
    mantyla, mantyla_source = _taxonomy_value_with_source(
        rule_id=rule_id,
        rule_map=MANTYLA_RULE_MAP,
        default="others",
    )

    return {
        "mantyla": mantyla,
        "_meta": {
            "version": CODE_SMELL_TAXONOMY_VERSION,
            "normalized_rule_id": _normalized(rule_id) or None,
            "sources": {
                "mantyla": mantyla_source,
            },
        },
    }
