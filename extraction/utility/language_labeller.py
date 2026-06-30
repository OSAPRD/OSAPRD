"""
Infer programming-language labels from file paths..
"""

from __future__ import annotations

from typing import Optional

EXTENSION_TO_LANGUAGE = {
    "py": "Python",
    "pyw": "Python",
    "java": "Java",
    "js": "JavaScript",
    "jsx": "JavaScript",
    "mjs": "JavaScript",
    "c": "C",
    "h": "C",
    "cpp": "C++",
    "cc": "C++",
    "cxx": "C++",
    "hpp": "C++",
    "hh": "C++",
    "hxx": "C++",
    "cs": "C#",
    "ts": "TypeScript",
    "tsx": "TypeScript",
    "rb": "Ruby",
    "go": "Go",
    "rs": "Rust",
    "php": "PHP",
    "swift": "Swift",
    "kt": "Kotlin",
    "kts": "Kotlin",
    "m": "Objective-C",
    "mm": "Objective-C++",
    "scala": "Scala",
    "sh": "Shell",
    "bash": "Shell",
    "zsh": "Shell",
    "ps1": "PowerShell",
    "sql": "SQL",
    "r": "R",
    "jl": "Julia",
    "lua": "Lua",
    "dart": "Dart",
    "json": "JSON",
    "yml": "YAML",
    "yaml": "YAML",
    "toml": "TOML",
    "ini": "INI",
    "md": "Markdown",
    "rst": "reStructuredText",
    "html": "HTML",
    "htm": "HTML",
    "css": "CSS",
    "scss": "SCSS",
    "sass": "Sass",
    "less": "Less",
    "xml": "XML",
    "vue": "Vue",
    "svelte": "Svelte",
    "ipynb": "Jupyter Notebook",
}


def infer_language(path: Optional[str]) -> Optional[str]:
    """
    Return a language label based on the final file extension.
    """
    if not path:
        return None
    name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." not in name:
        return None
    ext = name.rsplit(".", 1)[-1].lower()
    return EXTENSION_TO_LANGUAGE.get(ext)
