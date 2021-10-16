from datasets import load_dataset


class RawMTDataset(object):
    def __init__(self, dataset, target, source):
        self.data = dataset
        self.target = target
        self.source = source

    @classmethod
    def wmt14_en_de(cls):
        return cls(load_dataset("wmt14", "de-en"), "en", "de")

    @classmethod
    def wmt14_en_fr(cls):
        return cls(load_dataset("wmt14", "fr-en"), "en", "fr")

    @classmethod
    def wmt16_ro_en(cls):
        return cls(load_dataset("wmt16", "ro-en"), "ro", "en")


class TokenizedMTDataset(object):
    def __init__(
        self,
        dataset,
        tokenizer,
        pad_to_max=False,
        max_source_length=256,
        max_target_length=128,
        ignore_pad_for_loss=True,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer

        self._pad_to_max = pad_to_max
        self._max_source_length = max_source_length
        self._max_target_length = max_target_length
        self._ignore_pad_for_loss = ignore_pad_for_loss

    def preprocess(self, num_workers):
        column_names = self.dataset.data["train"].column_names
        return self.dataset.data.map(
            self._preprocess_fn,
            batched=True,
            num_proc=num_workers,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer",
        )

    def _preprocess_fn(self, examples):
        inputs = [ex[self.dataset.source] for ex in examples["translation"]]
        targets = [ex[self.dataset.target] for ex in examples["translation"]]
        # inputs = [prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(
            inputs, max_length=self._max_source_length, padding=self._pad_to_max, truncation=True
        )

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self._max_target_length,
                padding=self._pad_to_max,
                truncation=True,
            )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self._pad_to_max == "max_length" and self._ignore_pad_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
