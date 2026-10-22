#!/usr/bin/env bash
# Upload a prepared token dataset (shards + manifest.json) to Kaggle as a PRIVATE dataset.
#   scripts/run_container.sh scripts/kaggle_upload_data.sh data/nemotron_cc nanonemo-nemotron-cc
# Needs ~/.kaggle/kaggle.json on the host (Kaggle -> Settings -> API -> Create New Token).
set -euo pipefail

DATA_DIR=${1:?usage: kaggle_upload_data.sh <data_dir> <dataset-slug>}
SLUG=${2:?usage: kaggle_upload_data.sh <data_dir> <dataset-slug>}
USER_NAME=$(python -c "import json, os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")

# Upload only the shards and the manifest, never the raw downloads.
STAGE="$DATA_DIR/.kaggle_stage"
rm -rf "$STAGE" && mkdir -p "$STAGE"
ln "$DATA_DIR"/*.bin "$DATA_DIR"/manifest.json "$STAGE"/  # hard links: no extra disk space

cat > "$STAGE/dataset-metadata.json" <<META
{
  "title": "$SLUG",
  "id": "$USER_NAME/$SLUG",
  "licenses": [{"name": "other"}]
}
META

if kaggle datasets status "$USER_NAME/$SLUG" >/dev/null 2>&1; then
    kaggle datasets version -p "$STAGE" -m "Update shards" 
else
    kaggle datasets create -p "$STAGE"  # private unless --public is passed
fi
rm -rf "$STAGE"
