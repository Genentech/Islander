cd ${HOME}/Islander/src

export DATASET="lung"
echo "DATASET-${DATASET}_Islander"
python scBenchmarker.py \
    --islander \
    --saveadata \
    --dataset "${DATASET}" \
    --save_path "${HOME}/Islander/models/_${DATASET}_/MODE-mixup-ONLY_LEAK-16_MLP-128 128";
