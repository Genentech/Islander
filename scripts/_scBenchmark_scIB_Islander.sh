cd ${HOME}/Islander/src

# export DATASET_List=("heart" "eye" "gut_fetal" "skin" "COVID" "pancreas" \
#     "brain" "lung_fetal_donor" "lung_fetal_organoid" "breast" "lung")

# for DATASET in "${DATASET_List[@]}"; do
# echo -e "\n\n\n\n"
# echo "DATASET-${DATASET}_Islander"
# export CKPT_FOLDER="${HOME}/G-scIB_dev/models/_${DATASET}_/"
# for CKPT in ${CKPT_FOLDER}* ; do
# export CFG=$(echo ${CKPT} | awk -F'/' '{print $NF}')
# # echo ${CKPT}
# echo ${CFG}
# python scBenchmarker.py \
#     --islander \
#     --saveadata \
#     --obsm "Islander" \
#     --save_path "${CKPT}" \
#     --dataset "${DATASET}" \
#     --savecsv "${DATASET}_scIslander_New" ;
# done
# done

# export DATASET_List=("brain" "breast" "COVID")
export DATASET_List=("lung")

for DATASET in "${DATASET_List[@]}"; do
echo -e "\n\n\n\n"
echo "DATASET-${DATASET}_Islander"
python scBenchmarker.py \
    --islander \
    --saveadata \
    --dataset "${DATASET}" \
    --save_path "${HOME}/Islander/models/_${DATASET}_/MODE-mixup-ONLY_LEAK-16_MLP-128 128";
done
