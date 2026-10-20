# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f4d8e1264366940438f0353da6f289c7bef069d993d111f8106086ccd18c4a30 AS runtime-base

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        make \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG MODEL_URL=https://github.com/Timur713/hack_falcon_2026/releases/download/model-v1/vitb14_in280_ep100.safetensors
ARG MODEL_SHA256=645d7cbdd7b530958a66c95ac072419f3b3cdd903808997e26069def2fc70cb4
ARG MODEL_CONFIG_URL=https://github.com/Timur713/hack_falcon_2026/releases/download/model-v1/vitb14_in280_ep100.yaml
ARG MODEL_CONFIG_SHA256=ead3748ae7e8a13fe978a285cb3d647444070a71517bb412f88fd5718184dd7e
ARG VERIFIER_URL=https://github.com/Timur713/hack_falcon_2026/releases/download/model-v1/verifier_f0.pt
ARG VERIFIER_SHA256=52fc9b111351dbcd021ddf221b6546761e1609a96beeb5b9697c3e03585667c6

LABEL ai.deeptech.model.url="${MODEL_URL}" \
      ai.deeptech.model.sha256="${MODEL_SHA256}" \
      ai.deeptech.model.config_url="${MODEL_CONFIG_URL}" \
      ai.deeptech.model.config_sha256="${MODEL_CONFIG_SHA256}" \
      ai.deeptech.verifier.url="${VERIFIER_URL}" \
      ai.deeptech.verifier.sha256="${VERIFIER_SHA256}"

# Network access is allowed only while building the image.
COPY requirements.txt requirements-lock-cu121.txt ./
RUN python3 -m pip install --no-cache-dir --upgrade pip==24.3.1 \
    && python3 -m pip install --no-cache-dir --constraint requirements-lock-cu121.txt \
        --index-url "${PYTORCH_INDEX_URL}" \
        torch==2.5.1 torchvision==0.20.1 \
    && python3 -m pip install --no-cache-dir --constraint requirements-lock-cu121.txt --requirement requirements.txt \
    && python3 -m pip check

# Fail closed if code ever tries to install a dependency at runtime.
ENV PIP_NO_INDEX=1

# Explicit allowlist: never package the dataset, local experiments, credentials,
# or unrelated checkpoints from a developer's working directory.
COPY reid ./reid
COPY scripts ./scripts
COPY tests ./tests
COPY splits ./splits
COPY third_party/official ./third_party/official
COPY Makefile Dockerfile docker-compose.yml .dockerignore ./

# A clean clone downloads the fixed release archive at build time; private
# development can use the same SHA-verified local cache. The selected members,
# embedding dimension and postprocessing policy come from the pinned manifest,
# never from CLI defaults for the historical single-model stack.
FROM runtime-base AS fusion
ARG FUSION_BUNDLE=artifacts/fusion_bundle
ARG FUSION_ARCHIVE_CACHE=artifacts/fusion_bundle.tar.gz
ARG FUSION_ARCHIVE_URL=https://github.com/Timur713/hf_2026/releases/download/fusion-pair-d065-20260927-rc1/fusion-pair-d065-bundle.tar.gz
ARG FUSION_ARCHIVE_SHA256=9eeb99fae8bdcb228265ee83ac51da60da9896199d4746d843b30f3976fa49a8
ARG FUSION_MANIFEST_SHA256=4e3eacd2431ae634369b8e34689cb48d579f034207d1fd72bdff29661ad71e22
ARG FUSION_RELEASE_URL=https://github.com/Timur713/hf_2026/releases/tag/fusion-pair-d065-20260927-rc1
LABEL ai.deeptech.artifact.kind="fusion" \
      ai.deeptech.fusion.source_release="${FUSION_RELEASE_URL}" \
      ai.deeptech.fusion.manifest_sha256="${FUSION_MANIFEST_SHA256}" \
      ai.deeptech.fusion.archive_url="${FUSION_ARCHIVE_URL}" \
      ai.deeptech.fusion.archive_sha256="${FUSION_ARCHIVE_SHA256}" \
      ai.deeptech.model.url="" \
      ai.deeptech.model.sha256="" \
      ai.deeptech.model.config_url="" \
      ai.deeptech.model.config_sha256="" \
      ai.deeptech.verifier.url="" \
      ai.deeptech.verifier.sha256=""
RUN --mount=type=bind,target=/build-context \
    PYTHONPATH=. python3 scripts/stage_fusion_release.py \
        --cache-root /build-context --bundle "${FUSION_BUNDLE}" \
        --archive-cache "${FUSION_ARCHIVE_CACHE}" \
        --archive-url "${FUSION_ARCHIVE_URL}" \
        --archive-sha256 "${FUSION_ARCHIVE_SHA256}" \
        --manifest-sha256 "${FUSION_MANIFEST_SHA256}" \
        --source-release-url "${FUSION_RELEASE_URL}" \
        --destination /app/artifacts/models/fusion \
        --require-verifier \
    && PYTHONPATH=. python3 scripts/audit_release.py weights --root /app \
        --model /app/artifacts/models/fusion \
        --verifier /app/artifacts/models/fusion/verifier.pt \
        --require-all-data \
        --report /app/weights_manifest.json
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python3", "-m", "reid.infer.run"]
CMD ["--images", "/data/images", "--query", "/data/test_query.csv", "--gallery", "/data/test_gallery.csv", "--out", "/out", "--model", "/app/artifacts/models/fusion", "--verifier", "/app/artifacts/models/fusion/verifier.pt", "--twins", "off"]

# Historical model-v1 remains available only through --target legacy.
FROM runtime-base AS legacy
# Bind the context read-only, without copying unrelated local files. If the
# authenticated private-repo cache is absent, download fixed URLs during build.
# After the repository becomes public, clone -> docker build needs no manual fetch.
RUN --mount=type=bind,target=/build-context \
    python3 scripts/stage_release.py --cache-root /build-context \
        --model-url "${MODEL_URL}" --model-sha256 "${MODEL_SHA256}" \
        --config-url "${MODEL_CONFIG_URL}" --config-sha256 "${MODEL_CONFIG_SHA256}" \
        --verifier-url "${VERIFIER_URL}" --verifier-sha256 "${VERIFIER_SHA256}" \
    && test -f artifacts/models/final/weights.safetensors \
    && test -f artifacts/models/final/model_config.yaml \
    && test -f artifacts/models/verifier_f0/verifier.pt \
    && echo "${MODEL_SHA256}  artifacts/models/final/weights.safetensors" | sha256sum --check --strict \
    && echo "${MODEL_CONFIG_SHA256}  artifacts/models/final/model_config.yaml" | sha256sum --check --strict \
    && echo "${VERIFIER_SHA256}  artifacts/models/verifier_f0/verifier.pt" | sha256sum --check --strict \
    && PYTHONPATH=. python3 scripts/audit_release.py weights --root /app \
        --model /app/artifacts/models/final \
        --verifier /app/artifacts/models/verifier_f0/verifier.pt \
        --report /app/weights_manifest.json

# All assets are now inside the image; runtime must use network_mode: none.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENTRYPOINT ["python3", "-m", "reid.infer.run"]
CMD ["--images", "/data/images", "--query", "/data/test_query.csv", "--gallery", "/data/test_gallery.csv", "--out", "/out", "--model", "/app/artifacts/models/final", "--verifier", "/app/artifacts/models/verifier_f0/verifier.pt", "--twins", "off"]

# All-data D082 candidate, pinned to the immutable team release.
# Installation verifies the manifest, provenance and exact weight budget.
FROM runtime-base AS d082-production
ARG D082_ARCHIVE_URL=https://github.com/Timur713/hf_2026/releases/download/d082-candidate-20261019-r2/falcon-D082-legacy66-base-bundle.tar.gz
ARG D082_ARCHIVE_SHA256=0920e24524e9b32215323bbe3f8e47ae6e19d1f036b14eacfa596507b83d95d5
ARG D082_MANIFEST_SHA256=94125e2665558df43fb6701f7f0ca6b5c1cf5289cbfd48c83c03d1ee0180305c
ARG D082_RELEASE_URL=https://github.com/Timur713/hf_2026/releases/tag/d082-candidate-20261019-r2
LABEL ai.deeptech.artifact.kind="d082_production" \
      ai.deeptech.artifact.a5000_measurement="source-prototype-20261020" \
      ai.deeptech.extraction.runtime="fixed4-member-graphs-singleton-h2d" \
      ai.deeptech.d082.source_release="${D082_RELEASE_URL}" \
      ai.deeptech.d082.archive_url="${D082_ARCHIVE_URL}" \
      ai.deeptech.d082.archive_sha256="${D082_ARCHIVE_SHA256}" \
      ai.deeptech.d082.manifest_sha256="${D082_MANIFEST_SHA256}" \
      ai.deeptech.model.url="" ai.deeptech.model.sha256="" \
      ai.deeptech.model.config_url="" ai.deeptech.model.config_sha256="" \
      ai.deeptech.verifier.url="" ai.deeptech.verifier.sha256=""
RUN --mount=type=bind,target=/build-context \
    PYTHONPATH=. python3 scripts/stage_d082_production.py \
      --cache-root /build-context \
      --destination /app/artifacts/models/d082 \
      --archive-url "${D082_ARCHIVE_URL}" \
      --archive-sha256 "${D082_ARCHIVE_SHA256}" \
      --manifest-sha256 "${D082_MANIFEST_SHA256}" \
      --source-release-url "${D082_RELEASE_URL}"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
ENTRYPOINT ["python3", "-m", "reid.infer.d082_entrypoint"]
CMD ["--images", "/data/images", "--query", "/data/test_query.csv", "--gallery", "/data/test_gallery.csv", "--out", "/out", "--device", "cuda"]

# Development tests create temporary Git repositories. Keep their Git dependency
# in a dedicated QA target; inference remains independent of Git and network.
FROM d082-production AS qa
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Default submission: frozen D082 recipe and the output-preserving extractor.
# Measured A5000 source prototype and final integration checks are documented
# separately in docs/d082_acceptance.md.
FROM d082-production AS final
