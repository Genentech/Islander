cd ${HOME}/Islander/
mkdir -p data
export DATASET_List=("brain" "breast" "COVID" "eye" "gut_fetal" "heart" "lung" "lung_fetal_donor" "lung_fetal_organoid" "pancreas" "skin")
for DATASET in "${DATASET_List[@]}"; do
    mkdir -p  "data/${DATASET}"
done

wget -O data/brain/local.h5ad https://datasets.cellxgene.cziscience.com/b9171f05-8112-4a55-95f2-4cf8a57df8a2.h5ad 
wget -O data/breast/local.h5ad https://datasets.cellxgene.cziscience.com/b8b5be07-061b-4390-af0a-f9ced877a068.h5ad
wget -O data/COVID/su_2020_processed.HDF5 https://s3.us-west-2.amazonaws.com/atlas.fredhutch.org/data/hutch/covid19/downloads/su_2020_processed.HDF5
wget -O data/eye/local.h5ad https://datasets.cellxgene.cziscience.com/7440585d-c11d-448b-a91b-3a379d241d87.h5ad
wget -O data/gut_fetal/local.h5ad https://datasets.cellxgene.cziscience.com/5d27ffd6-1769-4564-961f-9bb32d9ca3a4.h5ad 
wget -O data/heart/local.h5ad https://datasets.cellxgene.cziscience.com/c1e3c998-4961-46c9-929d-d011900964e8.h5ad
wget -O data/lung/local.h5ad https://datasets.cellxgene.cziscience.com/b53b3bcd-3485-4562-a543-e473dfef3b27.h5ad
wget -O data/lung_fetal_donor/donor.h5ad https://datasets.cellxgene.cziscience.com/2b55f5c0-aa82-41e8-9d84-915b1d5a797b.h5ad
wget -O data/lung_fetal_organoid/organoid.h5ad https://datasets.cellxgene.cziscience.com/3d0778e7-bdaf-4a2c-a096-2c73522d8b45.h5ad
wget -O data/pancreas/human_pancreas_norm_complexBatch.h5ad https://figshare.com/ndownloader/files/24539828
wget -O data/skin/local.h5ad  https://datasets.cellxgene.cziscience.com/9d004f96-71ea-42dc-a2e6-e81e1bdedb48.h5ad