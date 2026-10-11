import pandas as pd
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator


RDLogger.DisableLog("rdApp.*")
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import argparse
from src.data.dataset import load_dataset
from src.data.features import compute_topo_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_smiles():
    url = "https://raw.githubusercontent.com/choderalab/nano-drugbank/master/df_drugbank_smiles.csv"
    logger.info(f"Downloading SMILES from {url}...")
    df = pd.read_csv(url)
    return (
        df[["drugbank_id", "smiles"]]
        .dropna()
        .set_index("drugbank_id")["smiles"]
        .to_dict()
    )


def get_rdkit_features(
    smiles: str,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:  # type: ignore
        return None, None


    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    fp_arr = np.zeros((0,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)


    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
    ]
    return fp_arr, np.array(desc, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--out_file", type=str, default="./data/ogbl_ddi/features.pt")
    args = parser.parse_args()


    data, _ = load_dataset(args.data_dir)
    num_nodes = data.num_nodes
    assert num_nodes is not None


    mapping_path = os.path.join(
        args.data_dir, "ogbl_ddi", "mapping", "nodeidx2drugid.csv.gz"
    )
    mapping_df = pd.read_csv(mapping_path)


    smiles_dict = download_smiles()

    morgan_list: list[np.ndarray] = []
    rdkit_list: list[np.ndarray] = []

    missing = 0
    for idx in range(num_nodes):
        drug_id = mapping_df[mapping_df["node idx"] == idx]["drug id"].iloc[0]
        smiles = smiles_dict.get(drug_id, None)

        fp, desc = None, None
        if smiles:
            fp, desc = get_rdkit_features(smiles)

        if fp is None or desc is None:
            missing += 1
            fp = np.zeros(2048, dtype=np.float32)
            desc = np.zeros(8, dtype=np.float32)

        morgan_list.append(fp)
        rdkit_list.append(desc)

    logger.info(f"Missing SMILES or invalid molecules: {missing} / {num_nodes}")

    morgan_feat = torch.tensor(np.stack(morgan_list))
    rdkit_feat = torch.tensor(np.stack(rdkit_list))


    rdkit_feat = (rdkit_feat - rdkit_feat.mean(dim=0)) / (rdkit_feat.std(dim=0) + 1e-8)


    assert data.edge_index is not None
    topo_feat = compute_topo_features(data.edge_index, num_nodes)

    features = {"morgan": morgan_feat, "rdkit": rdkit_feat, "topo": topo_feat}

    torch.save(features, args.out_file)
    logger.info(f"Features saved to {args.out_file}")


if __name__ == "__main__":
    main()
