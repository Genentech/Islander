cd ${HOME}/G-scIB_dev/src

# export DATASET_List=("skin" "lung" "lung_fetal_donor" "lung_fetal_organoid" \
#     "brain" "breast" "heart" "eye" "gut_fetal" "COVID" "pancreas")
# export DATASET_List=("lung_fetal_organoid" "lung_fetal_donor" "heart" "skin")
export DATASET_List=("gut_fetal")
export DATASET_List=("lung_fetal_organoid" "lung_fetal_donor" "heart")
for DATASET in "${DATASET_List[@]}"; do
echo -e "\n\n\n\n"
echo "DATASET-${DATASET}_Baselines"
python scBenchmarker.py \
    --obsm X_tsne X_pca X_umap \
    --tsne \
    --umap \
    --saveadata \
    --dataset "${DATASET}" \
    --savecsv "${DATASET}_Rerun" \
    --save_path "${HOME}/G-scIB_dev/models/_${DATASET}_/MODE-mixup-ONLY_LEAK-16_MLP-128 128";
done