"""
Tests for poetry SFT dataset construction and masked SFT samples.
@Author xiaomin.zhang
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from utils import Tokenizer


class PoetrySFTPipelineTest(unittest.TestCase):
    def test_classifies_and_formats_qiyan_jueju(self):
        from scripts.build_poetry_sft_dataset import build_instruction_sample, classify_poem

        poem = "男儿何不带吴钩，收取关山五十州。请君暂上凌烟阁，若个书生万户侯。"

        classification = classify_poem(poem)
        self.assertIsNotNone(classification)
        self.assertEqual(classification.poem_type, "七言绝句")
        self.assertEqual(
            classification.output,
            "男儿何不带吴钩，\n收取关山五十州。\n请君暂上凌烟阁，\n若个书生万户侯。",
        )

        sample = build_instruction_sample(poem, rng=random.Random(7), source_index=12)
        self.assertIsNotNone(sample)
        self.assertIn("七言绝句", sample["instruction"])
        self.assertEqual(sample["output"], classification.output)
        self.assertEqual(sample["metadata"]["poem_type"], "七言绝句")
        self.assertEqual(sample["metadata"]["source_index"], 12)

    def test_builds_multiple_constraint_instruction_samples(self):
        from scripts.build_poetry_sft_dataset import build_instruction_samples

        poem = "山色湖光入画屏，水云深处鸟声清。春风不尽江南意，一路花香到客亭。"

        samples = build_instruction_samples(
            poem,
            rng=random.Random(11),
            source_index=8,
            source_file="memory",
            max_samples_per_poem=8,
        )

        instruction_types = {sample["metadata"]["instruction_type"] for sample in samples}
        self.assertGreaterEqual(len(samples), 4)
        self.assertIn("basic", instruction_types)
        self.assertIn("keyword", instruction_types)
        self.assertIn("theme", instruction_types)
        self.assertIn("combined", instruction_types)

        keyword_samples = [
            sample for sample in samples
            if sample["metadata"]["instruction_type"] in {"keyword", "combined"}
        ]
        self.assertTrue(keyword_samples)
        for sample in keyword_samples:
            keywords = sample["metadata"]["constraint_keywords"]
            self.assertTrue(keywords)
            for keyword in keywords:
                self.assertIn(keyword, sample["instruction"])
                self.assertIn(keyword, sample["output"])

    def test_sft_dataset_masks_prompt_tokens(self):
        from train_poetry_sft import PoetrySFTDataset, format_sft_text

        example = {
            "instruction": "写一首五言绝句，主题是春天",
            "input": "",
            "output": "春风吹绿柳，\n花雨落前溪。",
            "metadata": {"poem_type": "五言绝句"},
        }
        formatted = format_sft_text(example)
        self.assertTrue(formatted.startswith("要求：写一首五言绝句"))
        self.assertIn("\n作品：", formatted)

        tokenizer = Tokenizer(vocab_size=200)
        tokenizer.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        for ch in sorted(set(formatted)):
            if ch not in tokenizer.word2idx:
                tokenizer.word2idx[ch] = len(tokenizer.word2idx)
        tokenizer.idx2word = {idx: token for token, idx in tokenizer.word2idx.items()}
        tokenizer._char_level = True

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.jsonl"
            path.write_text(json.dumps(example, ensure_ascii=False) + "\n", encoding="utf-8")

            dataset = PoetrySFTDataset(path, tokenizer, max_seq_len=80)
            input_ids, target_ids = dataset[0]

        self.assertEqual(input_ids.shape, target_ids.shape)
        self.assertEqual(input_ids.shape[0], 79)
        self.assertEqual(int(target_ids[0].item()), tokenizer.pad_id)
        self.assertGreater((target_ids != tokenizer.pad_id).sum().item(), 0)

    def test_early_stopping_respects_min_delta_and_patience(self):
        from train_poetry_sft import EarlyStopping

        early_stopping = EarlyStopping(patience=2, min_delta=0.01)

        self.assertFalse(early_stopping.update(3.0))
        self.assertEqual(early_stopping.best_loss, 3.0)
        self.assertEqual(early_stopping.bad_checks, 0)

        self.assertFalse(early_stopping.update(2.995))
        self.assertEqual(early_stopping.bad_checks, 1)

        self.assertTrue(early_stopping.update(2.996))
        self.assertEqual(early_stopping.bad_checks, 2)

        self.assertFalse(early_stopping.update(2.98))
        self.assertEqual(early_stopping.best_loss, 2.98)
        self.assertEqual(early_stopping.bad_checks, 0)


if __name__ == "__main__":
    unittest.main()
