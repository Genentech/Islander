cd ${HOME}/Islander/src

export LR=0.001
export EPOCH=10
export MODE="mixup"
export LEAKAGE_List=(16 64)
export MLP_List=("128 128")
# export MLP_List=("64 64" "128 128")
export DATASET_List=("lung" "lung_fetal_donor" "lung_fetal_organoid" \
    "brain" "breast" "heart" "eye" "gut_fetal" "skin" "COVID" "pancreas")

for DATASET in "${DATASET_List[@]}"; do
export PROJECT="_${DATASET}_"
export SavePrefix="${HOME}/Islander/models/${PROJECT}"
mkdir -p $SavePrefix
for MLPSIZE in "${MLP_List[@]}"; do
for LEAKAGE in "${LEAKAGE_List[@]}"; do
export RUNNAME="MODE-${MODE}-ONLY_LEAK-${LEAKAGE}_MLP-${MLPSIZE}"
echo "DATASET-${DATASET}_${RUNNAME}"
python scTrain.py \
    --gpu 3 \
    --lr ${LR} \
    --mode ${MODE} \
    --epoch ${EPOCH} \
    --w_rec 0 \
    --dataset ${DATASET} \
    --leakage ${LEAKAGE} \
    --project ${PROJECT} \
    --mlp_size ${MLPSIZE} \
    --runname "${RUNNAME}" \
    --savename "${SavePrefix}/${RUNNAME}" ;
done
done
done
