#!/usr/bin/env bash
##
## horus_agent
## Copyright (c) 2026 Temple Compute
##
## Commercial use and distribution of this software is prohibited.
## For commercial use, please contact the author for licensing options.
##

# Install horus-agent into the login container of horus-slurm's docker Slurm
# cluster (tests/integration/cluster there), under ITEST_DIR so the job user
# can run it. Usage: install_agent.sh <container> <itest_dir>
set -euo pipefail
container=$1
itest_dir=$2
repo=$(cd "$(dirname "$0")/../.." && pwd)
uid=$(id -u)

rm -rf "$itest_dir/dist"
(cd "$repo" && uv build --wheel -q -o "$itest_dir/dist")
docker cp "$(command -v uv)" "$container:/usr/local/bin/uv"
docker exec -u "$uid" -e HOME="$itest_dir" \
    -e UV_PYTHON_INSTALL_DIR="$itest_dir/python" \
    -e UV_CACHE_DIR="$itest_dir/uv-cache" \
    "$container" bash -c "
        set -e
        uv venv -q --clear --python 3.13 '$itest_dir/agent-venv'
        uv pip install -q --python '$itest_dir/agent-venv/bin/python' \
            '$itest_dir'/dist/*.whl
    "
