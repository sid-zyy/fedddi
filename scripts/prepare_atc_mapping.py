import os
import argparse
import pandas as pd
import torch
import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--out_file", type=str, default="./data/ogbl_ddi/atc_mapping.pt"
    )
    args = parser.parse_args()


    url = (
        "https://raw.githubusercontent.com/dhimmel/drugbank/gh-pages/data/drugbank.tsv"
    )
    logger.info(f"Downloading DrugBank mapping from {url}...")
    try:
        df = pd.read_csv(url, sep="\t")
    except Exception as e:
        logger.error(f"Failed to download or parse TSV: {e}")
        raise SystemExit(1)


    def extract_level1(atc_str: Any) -> str | None:
        if pd.isna(atc_str) or not isinstance(atc_str, str):
            return None
        codes: list[str] = atc_str.split("|")
        for code in codes:
            if len(code) > 0:
                return code[0].upper()
        return None

    df["atc_level1"] = df["atc_codes"].apply(extract_level1)
    db_to_atc = (
        df.dropna(subset=["atc_level1"])
        .set_index("drugbank_id")["atc_level1"]
        .to_dict()
    )


    mapping_path = os.path.join(
        args.data_dir, "ogbl_ddi", "mapping", "nodeidx2drugid.csv.gz"
    )
    if not os.path.exists(mapping_path):
        logger.error(
            f"Mapping file not found at {mapping_path}. "
            "The ogbl-ddi dataset needs to be downloaded first "
            "(e.g. `from src.data.dataset import load_dataset; load_dataset(root=%r)`) "
            "before this script can build the ATC mapping." % args.data_dir
        )
        raise SystemExit(1)
    mapping_df = pd.read_csv(mapping_path)

    num_nodes = mapping_df["node idx"].max() + 1



    valid_codes = ["A", "B", "C", "D", "G", "H", "J", "L", "M", "N", "P", "R", "S", "V"]
    code_to_idx = {code: i for i, code in enumerate(valid_codes)}
    UNKNOWN_IDX = len(valid_codes)

    atc_indices: list[int] = []
    missing = 0
    for idx in range(num_nodes):

        rows = mapping_df[mapping_df["node idx"] == idx]
        if len(rows) > 0:
            drug_id = rows.iloc[0]["drug id"]
            atc_code = db_to_atc.get(drug_id, None)
            if atc_code in code_to_idx:
                atc_indices.append(code_to_idx[atc_code])
            else:
                missing += 1
                atc_indices.append(UNKNOWN_IDX)
        else:
            missing += 1
            atc_indices.append(UNKNOWN_IDX)

    logger.info(f"Missing ATC codes or invalid nodes: {missing} / {num_nodes}")

    atc_tensor = torch.tensor(atc_indices, dtype=torch.long)
    torch.save(atc_tensor, args.out_file)
    logger.info(f"ATC mapping saved to {args.out_file}")


if __name__ == "__main__":
    main()
