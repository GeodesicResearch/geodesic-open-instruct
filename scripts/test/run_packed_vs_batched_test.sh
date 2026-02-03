#!/bin/bash
set -eo pipefail

source configs/beaker_configs/ray_node_setup.sh

PYTEST_EXIT=0
uv run pytest open_instruct/test_dpo_utils_gpu.py::TestPackedVsBatchedForward -xvs || PYTEST_EXIT=$?

exit $PYTEST_EXIT
