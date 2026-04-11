from pathlib import Path
import pandas as pd

IMAGE_ROOT = Path("/vol/biodata/data/Mammo/EMBED/pngs/1024x768")
INPUT_CSV = Path("EMBED_meta.csv")
OUTPUT_CSV = Path("EMBED_meta_cohort.csv")

df = pd.read_csv(INPUT_CSV, low_memory=False)

if "image_path" not in df.columns:
    raise ValueError("CSV must contain an 'image_path' column.")

df["image_path_orig"] = df["image_path"].astype(str)

def add_cohort_prefix(rel_path: str) -> str:
    rel_path = str(rel_path)

    if rel_path.startswith("cohort_1/") or rel_path.startswith("cohort_2/"):
        return rel_path

    p = Path(rel_path)
    parts = p.parts
    if len(parts) < 2:
        return rel_path

    patient_id = parts[0]
    filename = parts[-1]

    matches = []

    for cohort in ["cohort_1", "cohort_2"]:
        patient_root = IMAGE_ROOT / cohort / patient_id
        if not patient_root.exists():
            continue

        for match_path in patient_root.rglob(filename):
            matches.append((cohort, match_path))

    if len(matches) == 1:
        cohort, match_path = matches[0]
        rel_under_cohort = match_path.relative_to(IMAGE_ROOT / cohort)
        # print(f"Mapping {rel_path} to {cohort}/{rel_under_cohort.as_posix()}")
        return f"{cohort}/{rel_under_cohort.as_posix()}"

    if len(matches) == 0:
        print(f"Warning: No match found for {rel_path}")
        return rel_path

    raise ValueError(
        f"Ambiguous match for {rel_path}: "
        + ", ".join(
            f"{cohort}/{m.relative_to(IMAGE_ROOT / cohort).as_posix()}"
            for cohort, m in matches
        )
    )

df["image_path"] = df["image_path"].apply(add_cohort_prefix)

n_cohort1 = df["image_path"].str.startswith("cohort_1/").sum()
n_cohort2 = df["image_path"].str.startswith("cohort_2/").sum()
n_unchanged = (
    ~df["image_path"].str.startswith("cohort_1/")
    & ~df["image_path"].str.startswith("cohort_2/")
).sum()

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

print("Saved to:", OUTPUT_CSV)
print("Shape:", df.shape)
print("Rows mapped to cohort_1:", n_cohort1)
print("Rows mapped to cohort_2:", n_cohort2)
print("Rows left unchanged:", n_unchanged)
print(df[["image_path_orig", "image_path"]].head(20))