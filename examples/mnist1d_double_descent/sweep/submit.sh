#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
#
# Scheduler-neutral fan-out for the MNIST-1D double-descent sweep.
#
# This script generates one command per sweep cell and submits each through a
# site-supplied wrapper. Nothing about a specific scheduler, queue, host, module,
# or absolute path is baked in -- you provide that via environment variables:
#
#   OMNIBIAS_PYTHON   python interpreter to run the jobs        (default: python)
#   OMNIBIAS_SUBMIT   your cluster's submission wrapper prefix   (default: empty)
#                     e.g. a GPU batch submitter that takes the command as arguments.
#                     If empty, commands are printed instead of submitted (dry run).
#
# Any extra arguments are forwarded to gen_jobs.py (e.g. --arms cubic_newton
# --widths 40 50 60 --seeds 0 1 2 --noises 0.15 --scratch-base "artifacts/...").
#
# Example (dry run -- just prints the fan-out):
#   bash examples/mnist1d_double_descent/sweep/submit.sh --arms adam --widths 40 50
#
# Example (submit, wrapper supplied by your environment):
#   export OMNIBIAS_PYTHON=/path/to/venv/bin/python
#   export OMNIBIAS_SUBMIT='<your GPU batch submit command>'
#   bash examples/mnist1d_double_descent/sweep/submit.sh --arms cubic_newton --noises 0.15

set -euo pipefail

PYTHON="${OMNIBIAS_PYTHON:-python}"
SUBMIT="${OMNIBIAS_SUBMIT:-}"

mapfile -t JOBS < <("$PYTHON" -m examples.mnist1d_double_descent.sweep.gen_jobs --python "$PYTHON" "$@")

echo "Generated ${#JOBS[@]} jobs." >&2
for job in "${JOBS[@]}"; do
    [ -z "$job" ] && continue
    if [ -z "$SUBMIT" ]; then
        echo "$job"
    else
        echo "+ $SUBMIT $job" >&2
        eval "$SUBMIT $job"
    fi
done
