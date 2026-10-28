#!/usr/bin/env bash
# Download dataset sources (run on a machine that can reach Kaggle / Hugging Face). Use campus Wi-Fi.
# Every source must also exist in dataset.sources (db/ksl_bridge_v1.sql seeds them) with its licence.
set -euo pipefail
DATA=${DATA:-data/raw}
which="${1:-all}"

need() { command -v "$1" >/dev/null || { echo "missing $1 (pip install kaggle huggingface_hub)"; exit 1; }; }

if [[ $which == all || $which == ekitabu ]]; then
  need kaggle   # Kaggle → Account → Create API token → ~/.kaggle/kaggle.json
  kaggle datasets download -d ekitabu/ksl-video-dataset -p "$DATA/ekitabu-ksl-video" --unzip       # CC0, ~14.7 GB
fi
if [[ $which == all || $which == ksl-small ]]; then
  need kaggle
  kaggle datasets download -d joanwachuka/ksl-hand-landmarks -p "$DATA/ksl-hand-landmarks" --unzip
  kaggle datasets download -d michaelmwai/ksl-numpy-array-files-dataset -p "$DATA/ksl-numpy" --unzip
fi
if [[ $which == all || $which == asl-signs ]]; then
  need kaggle   # accept the competition rules on kaggle.com first; PRETRAIN ONLY
  kaggle competitions download -c asl-signs -p "$DATA/google-asl-signs"
fi
if [[ $which == all || $which == afrispeech ]]; then
  need huggingface-cli   # CC BY-NC-SA 4.0 → eval only
  huggingface-cli download intronhealth/afrispeech-200 --repo-type dataset --local-dir "$DATA/afrispeech-200"
fi
if [[ $which == all || $which == how2sign ]]; then
  need huggingface-cli   # pretrain only
  huggingface-cli download PSewmuthu/How2Sign_Holistic --repo-type dataset --local-dir "$DATA/how2sign-holistic"
fi
echo "Restricted sources (request access first): Maseno KSL pose (zenodo.org/records/14974973), AI4KSL corpus (arXiv 2410.18295)."
echo "Common Voice Swahili: download from Mozilla Data Collective (account required) into $DATA/common-voice-sw."
