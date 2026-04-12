from pathlib import Path
import os
from collections import defaultdict
import pandas as pd
from tqdm.auto import tqdm

IMAGE_ROOT = Path("/vol/biodata/data/Mammo/EMBED/pngs/1024x768")
INPUT_CSV = Path("EMBED_meta.csv")
OUTPUT_CSV = Path("EMBED_meta_cohort.csv")

df = pd.read_csv(INPUT_CSV, low_memory=False)

if "image_path" not in df.columns:
    raise ValueError("CSV must contain an 'image_path' column.")

df["image_path_orig"] = df["image_path"].astype(str)

# Build exactly the set of files we care about from the CSV:
# requested[patient_id] = {filename1, filename2, ...}
requested = defaultdict(set)

for rel_path in df["image_path_orig"]:
    rel_path = str(rel_path)
    p = Path(rel_path)
    parts = p.parts
    if len(parts) < 2:
        continue
    patient_id = parts[0]
    filename = parts[-1]
    requested[patient_id].add(filename)

lookup = {}
duplicates = []

for cohort in ["cohort_1", "cohort_2"]:
    cohort_root = IMAGE_ROOT / cohort

    for patient_id in tqdm(sorted(requested.keys()), desc=f"Indexing {cohort}"):
        patient_root = cohort_root / patient_id
        if not patient_root.exists():
            continue

        wanted_filenames = requested[patient_id]

        for root, _, files in os.walk(patient_root):
            root_path = Path(root)
            rel_root = root_path.relative_to(IMAGE_ROOT)

            for fname in files:
                if fname not in wanted_filenames:
                    continue

                key = (patient_id, fname)
                rel_path = (rel_root / fname).as_posix()

                if key in lookup and lookup[key] != rel_path:
                    duplicates.append((key, lookup[key], rel_path))
                else:
                    lookup[key] = rel_path

if duplicates:
    preview = duplicates[:10]
    raise ValueError(f"Found ambiguous matches, first few: {preview}")

missing = []
new_paths = []

for rel_path in tqdm(df["image_path"], total=len(df), desc="Rewriting image_path"):
    rel_path = str(rel_path)

    if rel_path.startswith("cohort_1/") or rel_path.startswith("cohort_2/"):
        new_paths.append(rel_path)
        continue

    p = Path(rel_path)
    parts = p.parts
    if len(parts) < 2:
        missing.append(rel_path)
        new_paths.append(rel_path)
        continue

    patient_id = parts[0]
    filename = parts[-1]

    out = lookup.get((patient_id, filename))
    if out is None:
        missing.append(rel_path)
        new_paths.append(rel_path)
    else:
        new_paths.append(out)

df["image_path"] = new_paths

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
print("Missing matches:", len(missing))
if missing:
    print("First 20 missing:")
    print(missing[:20])

print(df[["image_path_orig", "image_path"]].head(20))