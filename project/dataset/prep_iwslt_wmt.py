import random
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset

random.seed(42)


def main(iwslt_path, iwslt_dir, total_sample, save_dir):
    iwslt_dataset = load_dataset(iwslt_path, "de-en", iwslt_dir)
    print("Done loading iwslt dataset")
    wmt_dataset = load_dataset("wmt19", "de-en", split="train")
    print("Done loading wmt dataset")

    if total_sample <= 38_690_334:
        random_idx = random.sample(range(0, 38_690_334), total_sample - 160_239)
        random_idx = sorted(random_idx)
        sampled_dataset = wmt_dataset.select(random_idx)
        num_worker = 8
        print(f"Done sampling dataset with length {len(sampled_dataset)}")
    else:
        sampled_dataset = wmt_dataset
        num_worker = 16

    concat_ds = concatenate_datasets([iwslt_dataset["train"], sampled_dataset])
    concat_ds = concat_ds.shuffle(42)
    normalized_ds = concat_ds.map(
        lambda examples: {
            "translation": [
                {"en": ex["en"].lower(), "de": ex["de"].lower()} for ex in examples["translation"]
            ]
        },
        batched=True,
        remove_columns=["translation"],
        num_proc=num_worker,
    )
    print(f"Done combining both iwslt and wmt with length {len(normalized_ds)}")

    en_ds = normalized_ds.map(
        lambda examples: {"text": [ex["en"] for ex in examples["translation"]]},
        batched=True,
        remove_columns=["translation"],
        num_proc=num_worker,
    )
    en_lm_dataset = DatasetDict()
    en_lm_dataset["train"] = en_ds
    en_lm_dataset["validation"] = iwslt_dataset["validation"].map(
        lambda examples: {"text": [ex["en"] for ex in examples["translation"]]},
        batched=True,
        remove_columns=["translation"],
        num_proc=num_worker,
    )
    en_lm_dataset["test"] = iwslt_dataset["test"].map(
        lambda examples: {"text": [ex["en"] for ex in examples["translation"]]},
        batched=True,
        remove_columns=["translation"],
        num_proc=num_worker,
    )
    print(f"Done filtering english dataset with length {len(en_ds)}")
    en_lm_dataset.save_to_disk(f"{save_dir}_{total_sample}_en")

    de_ds = normalized_ds.map(
        lambda examples: {"text": [ex["de"] for ex in examples["translation"]]},
        batched=True,
        remove_columns=["translation"],
        num_proc=num_worker,
    )
    de_lm_dataset = DatasetDict()
    de_lm_dataset["train"] = de_ds
    de_lm_dataset["validation"] = iwslt_dataset["validation"].map(
        lambda examples: {"text": [ex["de"] for ex in examples["translation"]]},
        batched=True,
        remove_columns=["translation"],
        num_proc=num_worker,
    )
    de_lm_dataset["test"] = iwslt_dataset["test"].map(
        lambda examples: {"text": [ex["de"] for ex in examples["translation"]]},
        batched=True,
        remove_columns=["translation"],
        num_proc=num_worker,
    )
    print(f"Done filtering german dataset with length {len(de_ds)}")
    de_lm_dataset.save_to_disk(f"{save_dir}_{total_sample}_de")

    print("Saving mt dataset")
    mt_dataset = DatasetDict()
    mt_dataset["train"] = normalized_ds
    mt_dataset["validation"] = iwslt_dataset["validation"]
    mt_dataset["test"] = iwslt_dataset["test"]
    mt_dataset.save_to_disk(f"{save_dir}_{total_sample}")


if __name__ == "__main__":
    base_dir = "/storage/plzen1/home/akurniawan/adapters-project/dataset"
    iwslt_dir = base_dir + "/iwslt14"
    iwslt_path = iwslt_dir + "/iwslt_loader.py"
    wmt_dir = base_dir + "/wmt19_sample"
    total_sample = [500_000, 2_000_000, 40_000_000]
    for sample in total_sample:
        main(iwslt_path, iwslt_dir, sample, wmt_dir)
