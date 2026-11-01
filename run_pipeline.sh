#!/usr/bin/env bash
# Run the whole pipeline: build the index (if missing), answer, judge.
# Any extra arguments are passed to 02_run_agents.py, e.g.:
#   ./run_pipeline.sh --run quick-test --limit 30
#   ./run_pipeline.sh --run k20 --top-k 20
set -euo pipefail
cd "$(dirname "$0")"

# Use the project's virtualenv when there is one, even if it isn't activated
PY=python
[[ -x .venv/bin/python ]] && PY=.venv/bin/python

RUN="full"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  [[ "${args[i]}" == "--run" ]] && RUN="${args[i + 1]}"
done

if [[ -z "${CHROMA_HOST:-}" && ! -d chroma_db ]]; then
  echo "== 1/3 Building the Chroma index (one-off) =="
  $PY 01_build_financebench_index.py
fi

echo "== 2/3 Answering questions (run: $RUN) =="
$PY 02_run_agents.py "$@"

echo "== 3/3 Judging answers =="
$PY 03_evaluate_all_results.py --run "$RUN"

echo "Done. Compare runs with: mlflow ui --backend-store-uri sqlite:///mlflow.db"
