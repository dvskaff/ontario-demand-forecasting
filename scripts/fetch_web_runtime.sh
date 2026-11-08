#!/usr/bin/env bash
# Downloads ONNX Runtime Web from npm into web/vendor, so the weather what-if page serves the runtime itself
# instead of depending on a third-party CDN.
set -euo pipefail
VERSION="1.30.0"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-web/vendor}" in
  /*) DEST="$1" ;;
  *) DEST="$ROOT/${1:-web/vendor}" ;;
esac
mkdir -p "$DEST"
TMP="$(mktemp -d)"
cd "$TMP"
npm init -y > /dev/null
npm install --ignore-scripts --no-audit --no-fund "onnxruntime-web@${VERSION}" > /dev/null
for f in ort.wasm.min.mjs ort-wasm-simd-threaded.mjs ort-wasm-simd-threaded.wasm; do
  cp "node_modules/onnxruntime-web/dist/$f" "$DEST/"
done
ls -la "$DEST"
