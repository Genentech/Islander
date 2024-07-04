cd ${HOME}/Islander/src

export LR=0.001
export EPOCH=100
export MODE="triplet"
export LEAKAGE=16
export MLPSIZE="128 128"
export DATASET="lung_fetal_donor"
export SEED_List=(777 567 123)
# export DATASET_List=("brain" "lung_fetal_donor" "lung_fetal_organoid" "breast" "lung")

export DATASET_List=("lung" "lung_fetal_donor" "lung_fetal_organoid" \
    "brain" "breast" "heart" "eye" "gut_fetal" "skin" "COVID" "pancreas")

for SEED in "${SEED_List[@]}"; do
for DATASET in "${DATASET_List[@]}"; do
export PROJECT="_${DATASET}_"
export SavePrefix="${HOME}/Islander/models/${PROJECT}"
export RUNNAME="MODE-${MODE}-ONLY_LEAK-${LEAKAGE}_MLP-${MLPSIZE}_SEED-${SEED}"
echo "DATASET-${DATASET}_${RUNNAME}"
mkdir -p $SavePrefix
python scTrain.py \
    --gpu 4 \
    --lr ${LR} \
    --mode ${MODE} \
    --epoch ${EPOCH} \
    --w_rec 0 \
    --w_cet 1 \
    --seed ${SEED} \
    --dataset ${DATASET} \
    --leakage ${LEAKAGE} \
    --project ${PROJECT} \
    --mlp_size ${MLPSIZE} \
    --train_and_test True \
    --runname "${RUNNAME}" \
    --savename "${SavePrefix}/${RUNNAME}" ;
done
done