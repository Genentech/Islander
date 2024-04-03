# Ref: https://github.com/theislab/dca/blob/master/dca/io.py
from __future__ import division, print_function, absolute_import
import os, json, time, torch, shutil, random, pickle, numpy as np, scanpy as sc, Data_Handler as dh
from scipy.sparse import issparse, save_npz, load_npz, hstack, csr_matrix
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from typing import Union
from tqdm import tqdm

AnyRandom = Union[None, int, np.random.RandomState]
NumCells = 256  # 4096


def set_seed(seed_=0):
    random.seed(seed_)
    np.random.seed(seed_)
    torch.cuda.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True


# Ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class scDataset(Dataset):
    def __init__(
        self,
        random_state: AnyRandom = None,
        num_cells: int = NumCells,
        # full_train: bool = False,
        test_ratio: float = 0.1,
        inference: bool = False,
        rm_cache: bool = False,
        train_and_test=False,
        training: bool = True,
        batch_id: list = None,
        dataset: str = "lung",
        verbose: bool = True,
        n_genes: int = -1,
    ):
        self.train_and_test = train_and_test
        self.random_state = random_state
        self.test_ratio = test_ratio
        # self.full_train = full_train
        self.inference = inference
        self.training = training
        self.batch_id = batch_id
        self.n_cells = num_cells
        self.verbose = verbose
        self.dataset = dataset
        self.n_genes = n_genes
        # TODO: here can be optimized
        self._load_adata()

        if self.random_state:
            set_seed(self.random_state)

        if rm_cache:
            self._rmdirs()

        self.CELL2CAT = dh.CELL2CAT_[self.dataset]
        self.BATCH2CAT = dh.BATCH2CAT_[self.dataset]
        self.batch_key = dh.META_[self.dataset]["batch"]
        self.label_key = dh.META_[self.dataset]["celltype"]
        if self.verbose:
            print("\n" * 2)
            print("=" * 77)
            print("DATASET: %s" % (self.dataset))
            print("BATCH: %s, LABEL: %s" % (self.batch_key, self.label_key))
            print(f"Load {self.n_vars} Genes & {self.n_obs} Cells.")
            print("=" * 77 + "\n" * 2)
        if inference:
            self._outpath = join(dh.DATA_DIR, self.dataset, "Benchmark_")
            os.makedirs(self._outpath, exist_ok=True)
            if len(os.listdir(self._outpath)) < self.n_obs // self.n_cells:
                self._batchize_inference()

        elif self._check_presaved() and not rm_cache:
            _split = "Training" if self.training else "Testing"
            print("...Loading from Pre Saved, %s Split..." % _split)

        else:
            self._mkdirs()
            self._save_cfg()
            time_ = time.time()
            print("...Pre-Processing from Scratch...")
            self._count_quantified_batch()
            self._batchize()
            print("...Finished, Used %d Mins..." % ((time.time() - time_) // 60))
        self._split()
        del self.adata

    def _str_formatter(self, message):
        if self.verbose:
            print(f"\n=== {message} ===\n")

    def _check_presaved(self) -> bool:
        self._outpath = join(dh.DATA_DIR, self.dataset, "Train")
        if not exists(self._outpath):
            return False
        if (
            len(os.listdir(self._outpath))
            < (self.n_obs // self.n_cells) * (1 - self.test_ratio - 0.05) * 2
        ):
            return False
        return True

    def _save_cfg(self):
        _file = join(dh.DATA_DIR, self.dataset, "cfg.json")
        _cfgs = self.__dict__.copy()
        _cfgs["adata"] = None
        json.dump(_cfgs, open(_file, "w"))

    def _mkdirs(self):
        os.makedirs(join(dh.DATA_DIR, self.dataset), exist_ok=True)
        os.makedirs(join(dh.DATA_DIR, self.dataset, "Test"), exist_ok=True)
        os.makedirs(join(dh.DATA_DIR, self.dataset, "Train"), exist_ok=True)

    def _rmdirs(self):
        for _split in ["Train", "Test", "Benchmark_"]:
            if exists(join(dh.DATA_DIR, self.dataset, _split)):
                shutil.rmtree(join(dh.DATA_DIR, self.dataset, _split))

    def _load_adata(self):
        # ref: https://chat.openai.com/share/8bc3b625-1e97-4954-bde0-69306df2c062
        # normalize then select highly variable genes
        _suffix = "_hvg" if self.n_genes != -1 else ""
        adata = sc.read(dh.DATA_EMB_[self.dataset + _suffix])

        if self.n_genes != -1:
            sc.pp.highly_variable_genes(
                adata,
                subset=True,
                flavor="seurat_v3",
                n_top_genes=self.n_genes,
            )
        self.n_vars = adata.n_vars
        self.n_obs = adata.n_obs
        self.adata = adata
        return

    def _count_quantified_batch(self):
        count_ = 0
        for _id in self.adata.obs[self.batch_key].unique().to_list():
            _cells = (self.adata.obs[self.batch_key] == _id).sum()
            if _cells <= self.n_cells:
                continue
            count_ += 1
        self._str_formatter("Qualified batches: %d" % count_)

    @staticmethod
    def _isnan(value):
        return value != value

    def _conceptualize(self, adata, mask_tokens=["nan", "unknown"]):
        obj = dict()

        obj["cell_type"] = (
            adata.obs[self.label_key].apply(lambda x: self.CELL2CAT[x]).values.to_list()
        )
        obj["batch_id"] = (
            adata.obs[self.batch_key]
            .apply(lambda x: self.BATCH2CAT[x])
            .values.to_list()
        )

        return obj

    def _split(self):
        if self.inference:
            _split = "Benchmark_"
        else:
            _split = "Train" if self.training else "Test"
        _outpath = join(dh.DATA_DIR, self.dataset)
        _path = join(_outpath, _split)
        self.file_list = []

        for item in os.listdir(_path):
            self.file_list.append(
                join(_path, item.replace(".npz", "").replace(".pkl", ""))
            )
        # self.file_list = list(set(self.file_list))
        # if self.inference:
        #     self.file_list.sort()

        if self.train_and_test:
            _split = "Train" if _split == "Test" else "Test"
            _path = join(_outpath, _split)
            for item in os.listdir(_path):
                self.file_list.append(
                    join(_path, item.replace(".npz", "").replace(".pkl", ""))
                )
        self.file_list = list(set(self.file_list))
        if self.inference:
            self.file_list.sort()

    def _batchize(self):
        # ref: https://discuss.pytorch.org/t/best-way-to-load-a-lot-of-training-data/80847/2
        # Save files on disks instead of loading in MEM
        _path = join(dh.DATA_DIR, self.dataset)
        _num_cells = self.adata.__len__()
        _ids = np.arange(_num_cells)
        np.random.shuffle(_ids)

        _num_batch = round(_num_cells / self.n_cells)
        train_ids = np.ones(_num_batch)
        zero_indices = np.random.choice(
            _num_batch - 1, int(_num_batch * self.test_ratio), replace=False
        )
        train_ids[zero_indices] = 0
        train_ids[-1] = 0
        train_ids = train_ids.astype(bool)

        for _idx in tqdm(range(_num_batch)):
            adata_subset = self.adata[
                _ids[_idx * self.n_cells : (_idx + 1) * self.n_cells]
            ].copy()

            obj = self._conceptualize(adata_subset)
            _split = "Train" if train_ids[_idx] else "Test"
            _outfile = join(_path, _split, "_%s.pkl" % str(_idx).zfill(10))
            if isinstance(adata_subset.X, np.ndarray):
                adata_subset.X = csr_matrix(adata_subset.X)
            save_npz(_outfile.replace(".pkl", ".npz"), adata_subset.X)
            with open(_outfile, "wb") as handle:
                pickle.dump(obj, handle)

        # if self.full_train:
        #     for _f in os.listdir(join(_path, "Test")):
        #         shutil.copy(join(_path, "Test", _f), join(_path, "Train", _f))
        return

    def _batchize_inference(self):
        _num_cells = self.adata.__len__()
        _ids = np.arange(_num_cells)
        for _idx in tqdm(range(_num_cells // self.n_cells + 1)):
            adata_subset = self.adata[
                _ids[_idx * self.n_cells : (_idx + 1) * self.n_cells]
            ].copy()
            obj = self._conceptualize(adata_subset)
            _outfile = join(self._outpath, "_%s.pkl" % str(_idx).zfill(10))
            if isinstance(adata_subset.X, np.ndarray):
                adata_subset.X = csr_matrix(adata_subset.X)
            save_npz(_outfile.replace(".pkl", ".npz"), adata_subset.X)
            with open(_outfile, "wb") as handle:
                pickle.dump(obj, handle)
        return

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        infos = pickle.load(open(filename + ".pkl", "rb"))
        item = {"counts": torch.Tensor(load_npz(filename + ".npz").toarray())}
        item["cell_type"] = torch.Tensor(infos["cell_type"]).long()
        item["batch_id"] = torch.Tensor(infos["batch_id"])
        return item

    def sample_two_files(self):
        # Randomly sample two files from the list
        file_indices = random.sample(range(len(self.file_list)), 2)
        return [self.__getitem__(idx) for idx in file_indices]


def collate_fn(batch):
    output = {}
    for sample in batch:
        if isinstance(sample, dict):
            for key, value in sample.items():
                if (
                    isinstance(value, torch.Tensor)
                    or isinstance(value, np.ndarray)
                    or isinstance(value, list)
                ):
                    if key not in output:
                        output[key] = []
                    output[key].append(value)
                elif isinstance(value, dict):
                    if key not in output:
                        output[key] = {}
                    for k, v in value.items():
                        if k not in output[key]:
                            output[key][k] = []
                        output[key][k].append(v)
    for key, value in output.items():
        if isinstance(value, list):
            output[key] = torch.concat(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                output[key][k] = torch.tensor(v)
    return output


class TripletDataset(Dataset):
    def __init__(self, scDataset):
        self.file_list = scDataset.file_list

        self.labels = []
        for _file in self.file_list:
            infos = pickle.load(open(_file + ".pkl", "rb"))
            self.labels.append(infos["cell_type"])
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: torch.where(self.labels == label)[0] for label in self.labels_set
        }

    def __getitem__(self, index):
        anchor, label = self.file_list[index], self.labels[index]
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label])
        negative_label = np.random.choice(
            list(filter(lambda x: x != label, self.labels_set))
        )
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        positive, negative = (
            self.file_list[positive_index],
            self.file_list[negative_index],
        )
        return anchor, positive, negative

    def __len__(self):
        return len(self.file_list)


class ContrastiveDataset(Dataset):
    def __init__(self, scDataset):
        self.file_list = scDataset.file_list

        self.labels = []
        for _file in self.file_list:
            infos = pickle.load(open(_file + ".pkl", "rb"))
            self.labels.append(infos["cell_type"])
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: torch.where(self.labels == label)[0] for label in self.labels_set
        }

    def __getitem__(self, index):
        data1, label1 = self.file_list[index], self.labels[index]
        # Generate a positive example 50% of the time
        should_get_same_class = np.random.randint(0, 2)
        if should_get_same_class:
            while True:
                # Keep looping until the same class image is found
                idx2 = np.random.choice(self.label_to_indices[label1])
                if idx2 != index:
                    break
        else:
            label2 = np.random.choice(
                list(filter(lambda x: x != label1, self.labels_set))
            )
            idx2 = np.random.choice(self.label_to_indices[label2])

        data2 = self.file_list[idx2]
        return (data1, data2), should_get_same_class


class ContrastiveDatasetDUAL(Dataset):
    # in case there are many cell types (e.g., > 100)
    def __init__(self, scDataset):
        self.file_list = scDataset.file_list

    def __getitem__(self, index):
        file1 = self.file_list[index]
        data1 = torch.Tensor(load_npz(file1 + "_norm.npz").toarray())
        label1 = pickle.load(open(file1 + ".pkl", "rb"))["cell_type"]

        file2 = np.random.choice(self.file_list)
        data2 = torch.Tensor(load_npz(file2 + "_norm.npz").toarray())
        label2 = pickle.load(open(file2 + ".pkl", "rb"))["cell_type"]

        return data1, label1, data2, label2

    def __len__(self):
        return len(self.file_list)


if __name__ == "__main__":
    #
    for DATASET in [
        # "lung",
        # "lung_fetal_donor",
        # "lung_fetal_organoid",
        # "brain",
        # "breast",
        # "heart",
        # "eye",
        # "gut_fetal",
        # "skin",
        # "COVID",
        "pancreas",
    ]:
        scData_Train = scDataset(
            dataset=DATASET,
            rm_cache=True,
        )

        scData_TrainLoader = DataLoader(
            scData_Train,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )
        for id_, batch_ in enumerate(tqdm(scData_TrainLoader)):
            pass
            # print(
            #     id_,                              # Integer
            #     batch_["norm_counts"].size(),     # Tensor
            #     batch_["sample_id"].__len__(),    # List
            #     batch_["cell_type"].__len__(),    # Tensor
            # )
