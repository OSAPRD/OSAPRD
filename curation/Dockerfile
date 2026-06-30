FROM python:3.12-trixie@sha256:2575347025c314e37d89d4b353904edbe1824a6117b8eeffe52254879e4f6146

WORKDIR /curation

ARG DEBIAN_FRONTEND=noninteractive
ARG JRE22_URL="https://api.adoptium.net/v3/binary/version/jdk-22.0.2%2B9/linux/x64/jre/hotspot/normal/eclipse"
ARG JDK17_URL="https://api.adoptium.net/v3/binary/version/jdk-17.0.19%2B10/linux/x64/jdk/hotspot/normal/eclipse"
ARG REFACTORING_MINER_PP_REF="e49dcd2fac61f068a436ee52d5b66215561e2f40"
ARG PMD_VERSION="7.23.0"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    git-lfs \
    nodejs \
    npm \
    unzip \
    zip \
    patchelf \
    cmake \
    build-essential \
    ninja-build \
    clang \
    clang-tidy \
    cppcheck \
    llvm-19-dev \
    libclang-19-dev \
    && rm -rf /var/lib/apt/lists/*

# PMD supplies Java code-smell findings. DesigniteJava/DPy are licensed tools and
# are configured through mounted/local artifacts rather than downloaded here.
RUN curl -fsSL -o /tmp/pmd.zip "https://github.com/pmd/pmd/releases/download/pmd_releases%2F${PMD_VERSION}/pmd-dist-${PMD_VERSION}-bin.zip" \
    && unzip -q /tmp/pmd.zip -d /opt \
    && ln -sf "/opt/pmd-bin-${PMD_VERSION}/bin/pmd" /usr/local/bin/pmd \
    && chmod a+rx "/opt/pmd-bin-${PMD_VERSION}/bin/pmd" \
    && rm -f /tmp/pmd.zip

# ESLint supplies JavaScript code-smell findings. Version 8 keeps the
# --no-eslintrc CLI contract used for config-independent snapshot analysis.
RUN npm install -g eslint@8 \
    && npm cache clean --force

# Java 22 runs RefactoringMiner/RefDiff on modern Java repositories.
RUN mkdir -p /opt/jre22 \
    && curl -fsSL -o /tmp/jre22.tar.gz "${JRE22_URL}" \
    && tar -xzf /tmp/jre22.tar.gz -C /opt/jre22 --strip-components=1 \
    && rm -f /tmp/jre22.tar.gz

# Java 17 is used only while prebuilding RefactoringMiner++ in the image.
RUN mkdir -p /opt/jdk17 \
    && curl -fsSL -o /tmp/jdk17.tar.gz "${JDK17_URL}" \
    && tar -xzf /tmp/jdk17.tar.gz -C /opt/jdk17 --strip-components=1 \
    && rm -f /tmp/jdk17.tar.gz

COPY curation/requirements.txt /tmp/curation-requirements.txt
RUN python -m pip install --no-cache-dir pip==26.1.2 \
    && python -m pip install --no-cache-dir -r /tmp/curation-requirements.txt

ENV JAVA_HOME="/opt/jre22"
ENV PATH="/opt/jre22/bin:${PATH}"
RUN test -x /opt/jre22/bin/java \
    && ln -sf /opt/jre22/bin/java /usr/local/bin/java \
    && java -version

RUN mkdir -p /data/input /data/output

ENV CURATION_LOCAL_DIRECTORIES=/data/input
ENV CURATION_LOCAL_DATA_FORMAT=extractionpullrequests
ENV LOCAL_OUTPUT_DIR=/data/output

COPY . .

RUN find /curation -type d -exec chmod a+rx {} + \
    && find /curation -type f -exec chmod a+r {} +

RUN set -eux; \
    if [ -d /curation/tools/runtime ]; then \
    TOOL_BASE=/curation/tools; \
    elif [ -d /curation/curation/tools/runtime ]; then \
    TOOL_BASE=/curation/curation/tools; \
    else \
    echo "Unable to locate tools/runtime directory for RM++ installation." >&2; \
    exit 1; \
    fi; \
    JDK17_HOME=/opt/jdk17; \
    test -x "$JDK17_HOME/bin/java"; \
    TMP_DIR="$(mktemp -d)"; \
    RMPP_SRC="$TMP_DIR/RefactoringMinerPP"; \
    mkdir -p "$RMPP_SRC"; \
    curl -fsSL -o "$TMP_DIR/RefactoringMinerPP.tar.gz" "https://codeload.github.com/benzoinoo/RefactoringMinerPP/tar.gz/${REFACTORING_MINER_PP_REF}"; \
    tar -xzf "$TMP_DIR/RefactoringMinerPP.tar.gz" -C "$RMPP_SRC" --strip-components=1; \
    LLVM_MAJOR="$( (llvm-config --version 2>/dev/null || llvm-config-19 --version 2>/dev/null || echo 19) | cut -d. -f1 )"; \
    CAP_MAKEFILE="$RMPP_SRC/tools/cap/Makefile"; \
    if [ -f "$CAP_MAKEFILE" ]; then \
    sed -E -i "s|llvm-[0-9]+|llvm-${LLVM_MAJOR}|g; s|lclang-[0-9]+|lclang-${LLVM_MAJOR}|g" "$CAP_MAKEFILE"; \
    fi; \
    make -C "$RMPP_SRC/tools/cap"; \
    JAVA_HOME="$JDK17_HOME" PATH="$JDK17_HOME/bin:$PATH" "$RMPP_SRC/tools/RefactoringMiner/gradlew" --no-daemon -p "$RMPP_SRC/tools/RefactoringMiner" installDist -x test; \
    rm -rf "$TOOL_BASE/runtime/refactoringminerpp"; \
    cp -R "$RMPP_SRC/tools/RefactoringMiner/build/install/RefactoringMiner" "$TOOL_BASE/runtime/refactoringminerpp"; \
    cp "$RMPP_SRC/tools/cap/cap" "$TOOL_BASE/runtime/refactoringminerpp/bin/cap"; \
    chmod a+rx "$TOOL_BASE/runtime/refactoringminerpp/bin/RefactoringMiner"; \
    chmod a+rx "$TOOL_BASE/runtime/refactoringminerpp/bin/cap"; \
    rm -rf "$TMP_DIR"

RUN for TOOL_ROOT in /curation/tools /curation/curation/tools; do \
    if [ -d "$TOOL_ROOT" ]; then \
    chmod -R a+rX "$TOOL_ROOT"; \
    find "$TOOL_ROOT" -type f \( \
    -path '*/bin/*' -o \
    -name '*.sh' -o \
    -name '*.bash' -o \
    -name '*.py' -o \
    -name 'RefactoringMiner' -o \
    -name 'refdiff-example' \
    \) -exec chmod a+rx {} +; \
    fi; \
    done

RUN git lfs install --system \
    && python --version \
    && git --version \
    && java -version \
    && cmake --version \
    && clang --version \
    && clang-tidy --version \
    && cppcheck --version \
    && eslint --version \
    && pmd --version \
    && python -c "import pandas, pyarrow, multimetric" \
    && multimetric --help >/dev/null

# Patch the J2V8 native library inside RefDiff's JAR so Linux loads a non-execstack binary.
RUN J2V8_JAR=/curation/tools/runtime/refdiff/lib/j2v8_linux_x86_64-4.6.0.jar \
    && if [ ! -f "$J2V8_JAR" ]; then J2V8_JAR=/curation/curation/tools/runtime/refdiff/lib/j2v8_linux_x86_64-4.6.0.jar; fi \
    && TMP_DIR=$(mktemp -d) \
    && unzip -j "$J2V8_JAR" libj2v8_linux_x86_64.so -d "$TMP_DIR" \
    && if command -v execstack >/dev/null 2>&1; then \
    execstack -c "$TMP_DIR/libj2v8_linux_x86_64.so"; \
    elif patchelf --help 2>&1 | grep -q -- '--clear-execstack'; then \
    patchelf --clear-execstack "$TMP_DIR/libj2v8_linux_x86_64.so"; \
    else \
    echo "Skipping execstack cleanup: tool support not available in base image."; \
    fi \
    && cd "$TMP_DIR" \
    && zip -q -u "$J2V8_JAR" libj2v8_linux_x86_64.so \
    && rm -rf "$TMP_DIR"

ENTRYPOINT ["python", "-u", "-m", "curation.run"]
CMD ["--help"]
