cd ${HOME}/G-scIB_dev/src

# export DATASET_List=("skin" "lung" "lung_fetal_donor" "lung_fetal_organoid" \
#     "brain" "breast" "heart" "eye" "gut_fetal" "COVID" "pancreas")

export DATASET_List=("brain" "breast" "heart")
for DATASET in "${DATASET_List[@]}"; do
echo -e "\n\n\n\n"
echo "DATASET-${DATASET}_Baselines"
python scBenchmarker.py \
    --all \
    --saveadata \
    --dataset "${DATASET}" \
    --savecsv "${DATASET}_Baselines" \
    --save_path "${HOME}/G-scIB_dev/models/_${DATASET}_/MODE-mixup_LEAK-16_MLP-128 128";
done
