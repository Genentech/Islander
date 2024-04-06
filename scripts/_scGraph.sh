cd ${HOME}/Islander/src

export DATASET_List=("lung" "lung_fetal_donor" "lung_fetal_organoid" \
    "brain" "breast" "heart" "eye" "gut_fetal" "skin" "COVID" "pancreas")

export DATASET_List=("pancreas")

for DATASET in "${DATASET_List[@]}"; do
    echo "_${DATASET}_"
    python scGraph.py ${DATASET} ;
done
