cd ${HOME}/Islander/src

# export DATASET_List=("brain" "breast" "COVID")
export DATASET_List=("brain")
# export DATASET_List=("lung_fetal_donor")
for DATASET in "${DATASET_List[@]}"; do
echo -e "\n\n"
echo "DATASET-${DATASET}_Geneformer"
python scBenchmarker.py \
    --obsm_keys Geneformer \
    --dataset "${DATASET}" \
    --savecsv "${DATASET}_Geneformer_Board" \
    --save_path "${HOME}/Islander/models/_${DATASET}_/MODE-mixup-ONLY_LEAK-16_MLP-128 128";
done
