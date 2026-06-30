# Third-Party Tools and Licenses

This repository uses external tools for curation-time refactoring, maintainability, and code-smell analysis. Large tool binaries may be local-only, git-ignored, or built during Docker image creation. This notice records the active toolchain plus legacy tool references that are still represented by local vendor folders or compatibility mappings.

Important:

- License obligations still apply even when binaries are excluded from git.
- Verify the exact license text from the upstream project or local vendor package before redistributing a Docker image or packaged artifact.
- Keep this notice synchronized with `curation/Dockerfile`, `curation/requirements.txt`, `curation/config/refactoring_config.py`, and `curation/config/maintainability_config.py`.

## Active Curation Tools

### Refactoring Tools

1. RefactoringMiner
   - Purpose: mines refactorings for the original pull request before/after comparison.
   - Used by: Java and Python refactoring extraction paths.
   - Local paths: `curation/tools/runtime/refactoringminer/`, `curation/tools/bin/RefactoringMiner.bat`, and optional vendor snapshots such as `curation/tools/vendors/RefactoringMiner-3.1.3/`.
   - Upstream: <https://github.com/tsantalis/RefactoringMiner>
   - License: MIT. Check the upstream `LICENSE` file for the exact text.

2. RefDiff / ReffDiff wrapper
   - Purpose: mines JavaScript refactorings for the original pull request before/after comparison.
   - Used by: JavaScript refactoring extraction paths.
   - Local paths: `curation/tools/src/RefDiff/`, `curation/tools/runtime/refdiff/`, and `curation/tools/bin/ReffDiff.bat`.
   - Upstream: <https://github.com/aserg-ufmg/RefDiff>
   - License: MIT. The local source checkout includes `curation/tools/src/RefDiff/LICENSE`.

3. RefactoringMiner++
   - Purpose: mines C++ refactorings for the original pull request before/after comparison.
   - Used by: C++ refactoring extraction paths.
   - Local paths: `curation/tools/runtime/refactoringminerpp/` and `curation/tools/bin/RefactoringMinerPP.bat`.
   - Docker source: `curation/Dockerfile` builds from <https://github.com/benzoinoo/RefactoringMinerPP>, defaulting to commit `e49dcd2fac61f068a436ee52d5b66215561e2f40`.
   - License: MIT. Check the upstream `LICENSE` file for the exact text.

### Maintainability Tool

1. Multimetric
   - Purpose: computes maintainability metrics in the active curation pipeline.
   - Used by: `curation/metrics/maintainability_multimetric_metrics.py`.
   - Dependency: `multimetric==2.4.4` in `curation/requirements.txt`.
   - Upstream: <https://github.com/priv-kweihmann/multimetric>
   - License: Zlib for the installed `2.4.4` package metadata (`License-Expression: Zlib`, `LICENSE.Zlib`).
   - Note: duplicated-lines density is computed by this repository's custom deterministic duplication implementation, not by Multimetric.
   - Note: Multimetric emits maintainability measures, not code-smell instances. Code-smell instances are collected by the active code-smell tools below.

### Code-Smell Tools

1. DesigniteJava
   - Purpose: Java code-smell detection, including architectural, design, implementation, and ML-related smell categories.
   - Local path: `curation/tools/vendors/designitejava/DesigniteJava.jar` when present locally.
   - Upstream: <https://www.designite-tools.com/>
   - License: Designite commercial/proprietary terms. Verify the license from the downloaded package or vendor before use or redistribution.

2. DesignitePython / DPy
   - Purpose: Python code-smell detection, including architectural, design, implementation, testability, and test smells.
   - Historical distribution: DPy ZIP packages from Designite.
   - Upstream: <https://www.designite-tools.com/>
   - License: Designite commercial/proprietary terms. Verify the license from the downloaded package or vendor before use or redistribution.

3. PMD
   - Purpose: Java static analysis and smell-style findings.
   - Local path: `curation/tools/vendors/pmd-bin-7.23.0/` when present locally.
   - Docker source: `curation/Dockerfile` downloads PMD during image build.
   - Upstream: <https://pmd.github.io/>
   - License: see the local `curation/tools/vendors/pmd-bin-7.23.0/LICENSE` file when using the vendored copy. The distribution includes BSD-style PMD terms and bundled Apache-2.0 component notices.

4. ESLint
   - Purpose: JavaScript code-smell/static-analysis findings.
   - Local path: `curation/tools/runtime/nodetools/` when present locally.
   - Docker source: `curation/Dockerfile` installs `eslint@8` during image build.
   - Upstream: <https://eslint.org/>
   - License: MIT for ESLint. Follow installed npm package metadata for exact package and transitive dependency licenses.

5. Cppcheck
   - Purpose: C and C++ static analysis and smell-style findings.
   - Local path: `curation/tools/vendors/cppcheck/` when present locally.
   - Docker source: installed from the Linux package repository.
   - Upstream: <https://github.com/danmar/cppcheck>
   - License: GPL-3.0 as distributed in upstream `COPYING`, with additional component license files in local vendor packages where present.

6. clang-tidy and Clang Static Analyzer
   - Purpose: C and C++ code-smell/static-analysis mappings and diagnostics.
   - Current active use: Docker installs clang-tidy for smell detection and LLVM/Clang for RefactoringMiner++ support.
   - Local path: `curation/tools/vendors/llvm-20.1.8/` when present locally.
   - Upstream: <https://llvm.org/>
   - License: Apache-2.0 WITH LLVM-exception for LLVM project components. Verify local vendor notices for bundled dependencies.

### Active Docker Build and Runtime Toolchain

The curation Docker image installs build/runtime tools needed for the active curation pipeline and the C++ refactoring miner build:

- Eclipse Temurin JRE 22 and JDK 17 from Adoptium.
- LLVM/Clang packages, including `clang`, `clang-tidy`, `llvm-19-dev`, and `libclang-19-dev`.
- Cppcheck from the Linux package repository.
- PMD downloaded during Docker image build.
- Node.js/npm from the Linux package repository and ESLint installed through npm.
- Build and utility packages including `git`, `git-lfs`, `curl`, `unzip`, `zip`, `patchelf`, `cmake`, `build-essential`, and `ninja-build`.

These packages are distributed under their own upstream and Linux distribution licenses. Inspect the Docker image package metadata and upstream distributions before redistribution.

## Legacy and Compatibility Tool References

The current curation pipeline does not run SonarQube analysis, SonarJS analysis, static maintainability backends, or future-snapshot refactoring tool execution. The tools below are listed because local vendor artifacts, historical outputs, or compatibility mappings may still refer to them. Those mappings live primarily in `curation/config/code_smell_standardization_config.py` and `curation/config/code_smell_taxonomy_config.py`.

1. SonarJS rule mappings
   - Legacy purpose: JavaScript smell and rule normalization for historical SonarJS-backed outputs.
   - Upstream: <https://github.com/SonarSource/SonarJS>
   - License: follow the historical installed package metadata and lockfile for exact package versions and licenses.

2. Sonar Scanner, SonarQube, and SonarSource analyzers
   - Legacy purpose: Sonar-backed code-smell and static maintainability analysis.
   - Current status: not installed by the active curation Dockerfile and not used by the active curation pipeline.
   - Upstream docs: <https://docs.sonarsource.com/>
   - License: SonarSource distribution terms. Verify the exact product and analyzer licenses before use or redistribution.

## Maintainer Checklist

Before publishing an image, archive, or release:

1. Confirm every active binary and vendored package has a matching license notice.
2. Confirm local-only tools are either excluded from the artifact or documented with their license terms.
3. Re-check Docker-installed packages against the current `curation/Dockerfile`.
4. Re-check Python package licenses against `curation/requirements.txt` and installed package metadata.
5. Re-check legacy code-smell tool entries if compatibility mappings are removed or reactivated.
