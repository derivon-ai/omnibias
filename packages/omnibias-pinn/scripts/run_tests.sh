#!/usr/bin/env bash
# Local-only test runner for omnibias-pinn.
#
# Usage:
#   scripts/run_tests.sh [tier]
#
# Tiers (mode/scope):
#   fast    -- _core/, torch/, jax/ unit tests (default; runs on every change)
#   cross   -- adds tests/cross_backend/ bit-parity tests
#   integ   -- adds tests/integration/ parity-vs-research-code tests
#   full    -- everything (fast + cross + integration)
#
# Optional flags:
#   --backend=torch | jax | both       (default: both)
#   --python=$path-to-python           (default: $(which python))
#   --out-dir=$path                    (default: artifacts/omnibias-pinn-tests/<timestamp>)
#   -k <expr>                          forwarded to pytest
#   --                                 separator: pass-through args after this go to pytest
#
# Examples:
#   scripts/run_tests.sh fast
#   scripts/run_tests.sh cross --backend=both
#   scripts/run_tests.sh full -- -x --maxfail=3

set -euo pipefail

PKG_ROOT=$(cd "$(dirname "$0")/.." && pwd)
REPO_ROOT=$(cd "$PKG_ROOT/../.." && pwd)
TIER="${1:-fast}"
shift || true

BACKEND="both"
PY_BIN="$(which python)"
OUT_DIR=""
EXTRA_PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend=*)  BACKEND="${1#*=}"; shift ;;
    --python=*)   PY_BIN="${1#*=}"; shift ;;
    --out-dir=*)  OUT_DIR="${1#*=}"; shift ;;
    --) shift; EXTRA_PYTEST_ARGS=("$@"); break ;;
    *) EXTRA_PYTEST_ARGS+=("$1"); shift ;;
  esac
done

ts=$(date +%Y%m%d_%H%M%S)
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="artifacts/omnibias-pinn-tests/$ts"
fi
mkdir -p "$OUT_DIR"

case "$TIER" in
  fast)  PATHS=("tests/_core/")
         case "$BACKEND" in
           torch) PATHS+=("tests/torch/") ;;
           jax)   PATHS+=("tests/jax/") ;;
           both)  PATHS+=("tests/torch/" "tests/jax/") ;;
           *) echo "unknown --backend $BACKEND" >&2; exit 2 ;;
         esac ;;
  cross) PATHS=("tests/_core/" "tests/cross_backend/")
         case "$BACKEND" in
           torch) PATHS+=("tests/torch/") ;;
           jax)   PATHS+=("tests/jax/") ;;
           both)  PATHS+=("tests/torch/" "tests/jax/") ;;
         esac ;;
  integ) PATHS=("tests/_core/" "tests/integration/")
         case "$BACKEND" in
           torch) PATHS+=("tests/torch/") ;;
           jax)   PATHS+=("tests/jax/") ;;
           both)  PATHS+=("tests/torch/" "tests/jax/") ;;
         esac ;;
  full)  PATHS=("tests/")
         ;;
  *) echo "unknown tier $TIER (expected: fast|cross|integ|full)" >&2; exit 2 ;;
esac

echo "[omnibias-pinn] tier=$TIER  backend=$BACKEND  python=$PY_BIN"
echo "[omnibias-pinn] paths: ${PATHS[*]}"
echo "[omnibias-pinn] out-dir: $OUT_DIR"

cd "$PKG_ROOT"

JUNIT="$OUT_DIR/junit_${TIER}_${BACKEND}.xml"
LOG="$OUT_DIR/pytest_${TIER}_${BACKEND}.log"

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
  "$PY_BIN" -m pytest "${PATHS[@]}" \
    --junitxml="$JUNIT" \
    --tb=short \
    "${EXTRA_PYTEST_ARGS[@]}" 2>&1 | tee "$LOG"
exit_code=${PIPESTATUS[0]}

# Summary
"$PY_BIN" - "$JUNIT" "$OUT_DIR/metrics_${TIER}_${BACKEND}.json" "$exit_code" <<'PY'
import json, sys, xml.etree.ElementTree as ET
junit_path, metrics_path, exit_code = sys.argv[1], sys.argv[2], int(sys.argv[3])
metrics = {"exit_code": exit_code}
try:
    tree = ET.parse(junit_path)
    root = tree.getroot()
    suites = list(root) if root.tag == "testsuites" else [root]
    total = errors = failures = skipped = 0
    for s in suites:
        total += int(s.get("tests", 0))
        errors += int(s.get("errors", 0))
        failures += int(s.get("failures", 0))
        skipped += int(s.get("skipped", 0))
    metrics.update({
        "tests_total": total,
        "errors": errors,
        "failures": failures,
        "skipped": skipped,
        "passed": total - errors - failures - skipped,
        "pass": bool(errors == 0 and failures == 0),
    })
except Exception as e:
    metrics["junit_parse_error"] = repr(e)
    metrics["pass"] = (exit_code == 0)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2, sort_keys=True)
print(json.dumps(metrics, indent=2, sort_keys=True))
PY

exit $exit_code
