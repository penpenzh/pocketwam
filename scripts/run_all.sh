#!/usr/bin/env bash
export PYTHONUNBUFFERED=1   # stream logs
# PocketWAM full pipeline: collect -> pretrain -> train -> evaluate -> OOD eval.
# Stages whose artifacts already exist are SKIPPED, so re-running after an
# interruption is safe.
set -euo pipefail
cd "$(dirname "$0")/.."

CFG="configs/sft_wam_3m.yaml"
EXTRA_ARGS=${PWAM_EXTRA_ARGS:-}

collect_done()    { compgen -G "dataset/test.npz" >/dev/null && compgen -G "dataset/ood_test.npz" >/dev/null && compgen -G "dataset/pretrain_synth.npz" >/dev/null && compgen -G "dataset/pretrain_play.npz" >/dev/null; }
pretrain_done()   { compgen -G "outputs_pretrain_wam_3m/pocketwam-3m-pretrain/model.safetensors" >/dev/null; }
train_done()      { compgen -G "outputs_sft_wam_3m/train/pocketwam-3m/model.safetensors" >/dev/null; }
eval_done()       { compgen -G "outputs_sft_wam_3m/eval/eval_report.json" >/dev/null; }
ood_done()        { compgen -G "outputs_sft_wam_3m/eval_ood/eval_report.json" >/dev/null; }

if collect_done; then
    echo ">>> dataset already exists — skip collect"
else
    echo ">>> Stage 1/5: WAM dataset generation (SFT demos + pretrain set)"
    python3 -m pwam collect --config "${CFG}" ${EXTRA_ARGS}
fi

if pretrain_done; then
    echo ">>> Stage-0 pretrained backbone already exists — skip pretrain"
else
    echo ">>> Stage 2/5: Stage-0 world-model pretraining (video-only)"
    python3 -m pwam pretrain --config "${CFG}" ${EXTRA_ARGS}
fi

if train_done; then
    echo ">>> SFT checkpoint already exists — skip train"
else
    echo ">>> Stage 3/5: Stage-1 SFT (init from the pretrained backbone)"
    python3 -m pwam train --config "${CFG}" ${EXTRA_ARGS}
fi

if eval_done; then
    echo ">>> test-set eval already done — skip"
else
    echo ">>> Stage 4/5: closed-loop evaluation on the fixed 500-scenario test set"
    python3 -m pwam evaluate --config "${CFG}" ${EXTRA_ARGS}
fi

if ood_done; then
    echo ">>> OOD eval already done — skip"
else
    echo ">>> Stage 5/5: OOD generalization evaluation (300 scenarios, random 15 GIFs)"
    python3 -m pwam.eval_ood --config "${CFG}" --gifs 15 ${EXTRA_ARGS}
fi

echo ">>> PocketWAM pipeline complete."
