cd ${HOME}/Islander/src

# export DATASET_List=("skin" "lung" "lung_fetal_donor" "lung_fetal_organoid" \
#     "brain" "breast" "heart" "eye" "gut_fetal" "COVID" "pancreas")

export DATASET_List=("lung")
for DATASET in "${DATASET_List[@]}"; do
echo -e "\n\n"
echo "DATASET-${DATASET}_Baselines"
python scBenchmarker.py \
    --all \
    --dataset "${DATASET}" \
    --savecsv "${DATASET}_Baselines" \
    --save_path "${HOME}/Islander/models/_${DATASET}_/MODE-mixup-ONLY_LEAK-16_MLP-128 128"; 
    # --saveadata 
done
