export DATASET="lung"

cd ${HOME}/Islander/src
mkdir -p ${HOME}/Islander/res/scGraph

echo "_${DATASET}_"
python scGraph.py \
    --adata_path ${HOME}/Islander/data/${DATASET}/emb.h5ad \
    --batch_key sample \
    --label_key cell_type \
    --savename ${HOME}/Islander/res/scGraph/${DATASET};
