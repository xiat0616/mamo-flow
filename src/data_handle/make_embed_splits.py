import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_EMBED_ROOT = Path("/vol/biodata/data/Mammo/EMBED/")
DEFAULT_IMAGE_ROOT = DEFAULT_EMBED_ROOT / "pngs/1024x768"

DOMAIN_MAP = {
    "HOLOGIC, Inc.": 0,
    "GE MEDICAL SYSTEMS": 1,
    "FUJIFILM Corporation": 2,
    "GE HEALTHCARE": 3,
    "Lorad, A Hologic Company": 4,
}

TISSUE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}

MODELNAME_MAP = {
    "Selenia Dimensions": 0,
    "Senographe Essential VERSION ADS_53.40": 5,
    "Senographe Essential VERSION ADS_54.10": 5,
    "Senograph 2000D ADS_17.4.5": 2,
    "Senograph 2000D ADS_17.5": 2,
    "Lorad Selenia": 3,
    "Clearview CSm": 4,
    "Senographe Pristina": 1,
}


def normalize_domain_arg(domain):
    if domain in (None, "None"):
        return None
    if isinstance(domain, (list, tuple, set)):
        return [int(d) for d in domain]
    if isinstance(domain, str) and "," in domain:
        return [int(d.strip()) for d in domain.split(",")]
    return [int(domain)]


def normalize_scanner_model_arg(scanner_model):
    if scanner_model in (None, "None"):
        return None
    if isinstance(scanner_model, (list, tuple, set)):
        return [int(m) for m in scanner_model]
    if isinstance(scanner_model, str) and "," in scanner_model:
        return [int(m.strip()) for m in scanner_model.split(",")]
    return [int(scanner_model)]


def get_embed_df(
    csv_filepath: str,
    image_root: str | os.PathLike | None = None,
    exclude_cviews: bool = True,
    domain=None,
    scanner_model=None,
    hold_out_model_5: bool = True,
) -> pd.DataFrame:
    image_root = DEFAULT_IMAGE_ROOT if image_root is None else Path(image_root)
    df = pd.read_csv(csv_filepath, low_memory=False)

    df["shortpath"] = df["image_path"].astype(str)
    df["image_path"] = df["image_path"].apply(lambda x: str(image_root / str(x)))
    df["manufacturer_domain"] = df["Manufacturer"].map(DOMAIN_MAP)
    df["density"] = df["tissueden"].map(TISSUE_MAP)
    df["scanner"] = df["ManufacturerModelName"].map(MODELNAME_MAP)
    df["view"] = (df["ViewPosition"] != "MLO").astype(int)
    df["cview"] = (df["FinalImageType"] != "2D").astype(int)
    df["age"] = df["age_at_study"]

    df = df.dropna(
        subset=[
            "age",
            "density",
            "scanner",
            "view",
            "cview",
            "image_path",
            "empi_anon",
        ]
    ).copy()

    df["density"] = df["density"].astype(int)
    df["scanner"] = df["scanner"].astype(int)
    df["view"] = df["view"].astype(int)
    df["cview"] = df["cview"].astype(int)

    if exclude_cviews:
        df = df.loc[df["cview"] == 0].copy()

    if hold_out_model_5:
        df = df.loc[df["scanner"] != 5].copy()

    domains = normalize_domain_arg(domain)
    if domains is not None:
        df = df.loc[df["manufacturer_domain"].isin(domains)].copy()

        if DOMAIN_MAP["HOLOGIC, Inc."] in domains:
            hologic_mask = df["manufacturer_domain"] == DOMAIN_MAP["HOLOGIC, Inc."]
            df = df.loc[
                (~hologic_mask) | (df["ManufacturerModelName"] == "Selenia Dimensions")
            ].copy()

    scanner_models = normalize_scanner_model_arg(scanner_model)
    if scanner_models is not None:
        df = df.loc[df["scanner"].isin(scanner_models)].copy()

    return df


def split_df(
    df: pd.DataFrame,
    seed: int = 33,
    prop_train: float = 1.0,
    valid_frac: float = 0.125,
    test_frac: float = 0.125,
) -> dict[str, pd.DataFrame]:
    if not (0.0 <= valid_frac < 1.0):
        raise ValueError(f"valid_frac must be in [0, 1), got {valid_frac}")
    if not (0.0 <= test_frac < 1.0):
        raise ValueError(f"test_frac must be in [0, 1), got {test_frac}")
    if valid_frac + test_frac >= 1.0:
        raise ValueError(
            f"valid_frac + test_frac must be < 1, got {valid_frac + test_frac}"
        )

    patient_ids = df["empi_anon"].unique()

    train_frac = 1.0 - valid_frac - test_frac
    holdout_frac = valid_frac + test_frac

    train_id, holdout_id = train_test_split(
        patient_ids,
        train_size=train_frac,
        random_state=seed,
    )

    if prop_train < 1.0:
        train_id = np.sort(train_id)
        y = (
            df.loc[df["empi_anon"].isin(train_id)]
            .groupby("empi_anon")["scanner"]
            .unique()
            .apply(lambda x: x[0])
            .sort_index()
        )
        assert y.index[0] == train_id[0]
        train_id, _ = train_test_split(
            train_id,
            train_size=prop_train,
            stratify=y.values,
            random_state=seed,
        )

    valid_frac_of_holdout = valid_frac / holdout_frac
    valid_id, test_id = train_test_split(
        holdout_id,
        train_size=valid_frac_of_holdout,
        random_state=seed,
    )

    return {
        "train": df.loc[df["empi_anon"].isin(train_id)].copy(),
        "valid": df.loc[df["empi_anon"].isin(valid_id)].copy(),
        "test": df.loc[df["empi_anon"].isin(test_id)].copy(),
    }


def add_cache_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    if "cache_idx" in df.columns:
        df = df.drop(columns=["cache_idx"])
    df.insert(0, "cache_idx", np.arange(len(df), dtype=np.int64))
    return df


def save_split_csvs(
    split_dfs: dict[str, pd.DataFrame],
    out_dir: str | os.PathLike,
    overwrite: bool = False,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_paths = {}
    for split, df in split_dfs.items():
        path = out_dir / f"{split}.csv"
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite 1 to replace it."
            )
        df.to_csv(path, index=False)
        out_paths[split] = str(path)

    return out_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--csv_filepath", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--domain", type=str, nargs="+", default=None)
    parser.add_argument("--scanner_model", type=str, nargs="+", default=None)
    parser.add_argument("--exclude_cviews", type=int, default=1)
    parser.add_argument("--hold_out_model_5", type=int, default=1)

    parser.add_argument("--prop_train", type=float, default=1.0)
    parser.add_argument("--valid_frac", type=float, default=0.125)
    parser.add_argument("--test_frac", type=float, default=0.125)
    parser.add_argument("--split_seed", type=int, default=33)

    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()

    df = get_embed_df(
        csv_filepath=args.csv_filepath,
        image_root=args.data_dir,
        exclude_cviews=bool(args.exclude_cviews),
        domain=args.domain,
        scanner_model=args.scanner_model,
        hold_out_model_5=bool(args.hold_out_model_5),
    )

    split_dfs = split_df(
        df=df,
        seed=args.split_seed,
        prop_train=args.prop_train,
        valid_frac=args.valid_frac,
        test_frac=args.test_frac,
    )

    split_dfs = {k: add_cache_idx(v) for k, v in split_dfs.items()}

    out_paths = save_split_csvs(
        split_dfs=split_dfs,
        out_dir=args.out_dir,
        overwrite=bool(args.overwrite),
    )

    meta = {
        "csv_filepath": args.csv_filepath,
        "data_dir": args.data_dir,
        "domain": args.domain,
        "scanner_model": args.scanner_model,
        "exclude_cviews": int(args.exclude_cviews),
        "hold_out_model_5": int(args.hold_out_model_5),
        "prop_train": float(args.prop_train),
        "valid_frac": float(args.valid_frac),
        "test_frac": float(args.test_frac),
        "split_seed": int(args.split_seed),
        "counts": {k: int(len(v)) for k, v in split_dfs.items()},
    }

    meta_path = Path(args.out_dir) / "split_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved split CSVs:")
    for k, v in out_paths.items():
        print(f"  {k}: {v}")
    print(f"  meta: {meta_path}")

    overlap_train_valid = set(split_dfs["train"]["empi_anon"]).intersection(
        set(split_dfs["valid"]["empi_anon"])
    )
    overlap_train_test = set(split_dfs["train"]["empi_anon"]).intersection(
        set(split_dfs["test"]["empi_anon"])
    )
    overlap_valid_test = set(split_dfs["valid"]["empi_anon"]).intersection(
        set(split_dfs["test"]["empi_anon"])
    )

    print("\nPatient overlap check:")
    print(f"  train ∩ valid: {len(overlap_train_valid)}")
    print(f"  train ∩ test : {len(overlap_train_test)}")
    print(f"  valid ∩ test : {len(overlap_valid_test)}")


if __name__ == "__main__":
    main()