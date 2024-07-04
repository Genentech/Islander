cd ${HOME}/Islander/src

export LR=0.001
export EPOCH=10
export LEAKAGE=16
export MODE="mixup"
export MLPSIZE="128 128"
export DATASET_List=("lung" "lung_fetal_donor" "lung_fetal_organoid" \
    "brain" "breast" "heart" "eye" "gut_fetal" "skin" "COVID" "pancreas")

for DATASET in "${DATASET_List[@]}"; do
export PROJECT="_${DATASET}_"
export SavePrefix="${HOME}/Islander/models/${PROJECT}"
export RUNNAME="MODE-${MODE}-ONLY_LEAK-${LEAKAGE}_MLP-${MLPSIZE}"
echo "DATASET-${DATASET}_${RUNNAME}"
mkdir -p $SavePrefix
python scTrain.py \
    --gpu 3 \
    --lr ${LR} \
    --mode ${MODE} \
    --epoch ${EPOCH} \
    --dataset ${DATASET} \
    --leakage ${LEAKAGE} \
    --project ${PROJECT} \
    --mlp_size ${MLPSIZE} \
    --runname "${RUNNAME}" \
    --savename "${SavePrefix}/${RUNNAME}" ;
done