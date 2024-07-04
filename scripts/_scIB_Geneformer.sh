cd ${HOME}/Islander/src

export DATASET="brain"
echo -e "\n\n"
echo "DATASET-${DATASET}_Geneformer"
python scBenchmarker.py \
    --obsm_keys Geneformer \
    --dataset "${DATASET}" \
    --savecsv "${DATASET}_Geneformer" \
    --save_path "${HOME}/Islander/models/_${DATASET}_/MODE-mixup-ONLY_LEAK-16_MLP-128 128";
