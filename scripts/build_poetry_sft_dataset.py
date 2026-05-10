"""
Build a compact poetry instruction-tuning dataset from processed poetry text.
@Author xiaomin.zhang
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TERMINAL_PUNCT = "，。！？；"
NOISE_PATTERN = re.compile(r"[A-Za-z0-9<>《》\[\]{}（）()“”\"'「」『』〈〉【】·—…]")
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")

FORM_NAMES = {
    (4, 5): "五言绝句",
    (4, 7): "七言绝句",
    (8, 5): "五言律诗",
    (8, 7): "七言律诗",
}

SEASON_RULES = [
    ("春", ["春"]),
    ("夏", ["夏"]),
    ("秋", ["秋"]),
    ("冬", ["冬"]),
]

THEME_RULES = [
    ("山水", ["山", "水", "江", "湖", "溪", "泉", "峰"], 2),
    ("月夜", ["月", "夜"], 1),
    ("花草", ["花", "柳", "草", "桃", "梅", "菊", "竹"], 2),
    ("边塞", ["边", "塞", "关", "胡", "马", "烽", "戍", "沙"], 2),
    ("田园", ["田", "园", "桑", "禾", "农", "村", "柴"], 2),
    ("离别", ["别", "离", "送"], 1),
    ("思乡", ["乡", "故园", "故山"], 1),
    ("饮酒", ["酒", "杯", "醉", "樽"], 1),
    ("怀古", ["古", "台", "宫", "陵", "碑"], 2),
]

EMOTION_RULES = [
    ("思乡", ["乡", "故园", "故山", "归", "客"]),
    ("离别", ["别", "离", "送", "泪"]),
    ("闲适", ["闲", "眠", "坐", "卧", "悠"]),
    ("壮志", ["剑", "马", "关", "万里", "丈夫"]),
    ("伤怀", ["愁", "恨", "泪", "悲", "怅"]),
]

STYLE_RULES = [
    ("豪放", ["剑", "马", "关", "万里", "天地", "丈夫", "酒"]),
    ("婉约", ["月", "梦", "泪", "愁", "香", "罗"]),
    ("清新", ["春", "花", "柳", "鸟", "水", "云", "山"]),
]

KEYWORD_CANDIDATES = [
    "春", "夏", "秋", "冬", "月", "风", "雨", "云", "山", "水",
    "江", "花", "柳", "雪", "梅", "竹", "酒", "舟", "客", "雁",
    "马", "梦", "愁", "归", "夜", "天", "日", "海",
]

INSTRUCTION_TYPE_TARGETS = {
    "keyword": 0.35,
    "combined": 0.30,
    "theme": 0.20,
    "basic": 0.10,
    "season": 0.05,
}


@dataclass(frozen=True)
class PoemClassification:
    poem_type: str
    output: str
    line_count: int
    line_char_count: int
    season: str
    theme: str
    emotion: str
    style: str
    keywords: List[str]
    difficulty: str


def normalize_poem(text: str) -> str:
    """Normalize punctuation and remove whitespace."""
    text = text.strip()
    text = text.replace(",", "，").replace(".", "。")
    text = text.replace("!", "！").replace("?", "？")
    text = text.replace(";", "；").replace(":", "：")
    return re.sub(r"\s+", "", text)


def split_poem_units(text: str) -> List[str]:
    """Split a one-line poem into punctuation-preserving poetic units."""
    text = normalize_poem(text)
    units: List[str] = []
    current: List[str] = []
    for char in text:
        current.append(char)
        if char in TERMINAL_PUNCT:
            unit = "".join(current).strip()
            if unit:
                units.append(unit)
            current = []
    if current:
        unit = "".join(current).strip()
        if unit:
            units.append(unit)
    return units


def count_chinese(text: str) -> int:
    return len(CHINESE_PATTERN.findall(text))


def contains_noise(text: str) -> bool:
    return bool(NOISE_PATTERN.search(text))


def infer_tag(text: str, rules: Iterable[tuple[str, List[str]]], default: str = "") -> str:
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return default


def infer_theme(text: str) -> str:
    for label, keywords, min_matches in THEME_RULES:
        matches = sum(1 for keyword in keywords if keyword in text)
        if matches >= min_matches:
            return label
    return ""


def infer_keywords(text: str, limit: int = 2) -> List[str]:
    found = [keyword for keyword in KEYWORD_CANDIDATES if keyword in text]
    return found[:limit]


def classify_poem(poem: str) -> Optional[PoemClassification]:
    """Return a strict classical-form classification, or None if unsuitable."""
    poem = normalize_poem(poem)
    if not poem or contains_noise(poem) or not CHINESE_PATTERN.search(poem):
        return None

    units = split_poem_units(poem)
    if len(units) not in (4, 8):
        return None

    line_lengths = [count_chinese(unit) for unit in units]
    if len(set(line_lengths)) != 1:
        return None

    line_count = len(units)
    line_char_count = line_lengths[0]
    poem_type = FORM_NAMES.get((line_count, line_char_count))
    if poem_type is None:
        return None

    output = "\n".join(units)
    compact_text = re.sub(rf"[{TERMINAL_PUNCT}\n]", "", output)
    if len(compact_text) < 20 or len(output) > 96:
        return None

    keywords = infer_keywords(output)
    season = infer_tag(output, SEASON_RULES)
    theme = infer_theme(output)
    emotion = infer_tag(output, EMOTION_RULES)
    style = infer_tag(output, STYLE_RULES, default="古典")
    difficulty = "高级" if line_count == 8 else "中级"

    return PoemClassification(
        poem_type=poem_type,
        output=output,
        line_count=line_count,
        line_char_count=line_char_count,
        season=season,
        theme=theme,
        emotion=emotion,
        style=style,
        keywords=keywords,
        difficulty=difficulty,
    )


def instruction_variants(classification: PoemClassification) -> List[Dict]:
    """Build short, verifiable instruction variants for one classified poem."""
    variants = [
        {
            "instruction": f"写一首{classification.poem_type}。",
            "instruction_type": "basic",
            "constraint_keywords": [],
            "constraint_theme": "",
            "constraint_season": "",
        },
        {
            "instruction": f"请创作一首{classification.poem_type}。",
            "instruction_type": "basic",
            "constraint_keywords": [],
            "constraint_theme": "",
            "constraint_season": "",
        },
        {
            "instruction": f"请写一首符合{classification.poem_type}形式的作品。",
            "instruction_type": "basic",
            "constraint_keywords": [],
            "constraint_theme": "",
            "constraint_season": "",
        },
    ]
    if classification.theme:
        variants.extend([
            {
                "instruction": f"写一首{classification.theme}题材的{classification.poem_type}。",
                "instruction_type": "theme",
                "constraint_keywords": [],
                "constraint_theme": classification.theme,
                "constraint_season": "",
            },
            {
                "instruction": f"请以{classification.theme}为题创作一首{classification.poem_type}。",
                "instruction_type": "theme",
                "constraint_keywords": [],
                "constraint_theme": classification.theme,
                "constraint_season": "",
            },
        ])
    if classification.season:
        variants.append({
            "instruction": f"写一首带有{classification.season}天意象的{classification.poem_type}。",
            "instruction_type": "season",
            "constraint_keywords": [],
            "constraint_theme": "",
            "constraint_season": classification.season,
        })
    if classification.keywords:
        quoted = "、".join(f"“{keyword}”" for keyword in classification.keywords)
        variants.extend([
            {
                "instruction": f"写一首{classification.poem_type}，包含{quoted}。",
                "instruction_type": "keyword",
                "constraint_keywords": classification.keywords,
                "constraint_theme": "",
                "constraint_season": "",
            },
            {
                "instruction": f"请创作一首包含{quoted}的{classification.poem_type}。",
                "instruction_type": "keyword",
                "constraint_keywords": classification.keywords,
                "constraint_theme": "",
                "constraint_season": "",
            },
        ])
    if classification.theme and classification.keywords:
        quoted = "、".join(f"“{keyword}”" for keyword in classification.keywords)
        variants.extend([
            {
                "instruction": f"写一首{classification.theme}题材的{classification.poem_type}，包含{quoted}。",
                "instruction_type": "combined",
                "constraint_keywords": classification.keywords,
                "constraint_theme": classification.theme,
                "constraint_season": "",
            },
            {
                "instruction": f"请以{classification.theme}为题写一首{classification.poem_type}，包含{quoted}。",
                "instruction_type": "combined",
                "constraint_keywords": classification.keywords,
                "constraint_theme": classification.theme,
                "constraint_season": "",
            },
        ])
    if classification.season and classification.keywords:
        quoted = "、".join(f"“{keyword}”" for keyword in classification.keywords)
        variants.append({
            "instruction": f"写一首带有{classification.season}天意象的{classification.poem_type}，包含{quoted}。",
            "instruction_type": "combined",
            "constraint_keywords": classification.keywords,
            "constraint_theme": "",
            "constraint_season": classification.season,
        })
    return variants


def instruction_templates(classification: PoemClassification) -> List[str]:
    return [variant["instruction"] for variant in instruction_variants(classification)]


def build_metadata(
    classification: PoemClassification,
    source_index: int,
    source_file: str,
    variant: Dict,
) -> Dict:
    metadata = asdict(classification)
    metadata.pop("output")
    metadata["source_index"] = source_index
    if source_file:
        metadata["source_file"] = source_file
    metadata["instruction_type"] = variant["instruction_type"]
    metadata["constraint_keywords"] = variant["constraint_keywords"]
    metadata["constraint_theme"] = variant["constraint_theme"]
    metadata["constraint_season"] = variant["constraint_season"]
    return metadata


def build_instruction_sample(
    poem: str,
    rng: random.Random,
    source_index: int,
    source_file: str = "",
) -> Optional[Dict]:
    samples = build_instruction_samples(poem, rng, source_index, source_file, max_samples_per_poem=1)
    if not samples:
        return None
    return samples[0]


def build_instruction_samples(
    poem: str,
    rng: random.Random,
    source_index: int,
    source_file: str = "",
    max_samples_per_poem: int = 4,
) -> List[Dict]:
    classification = classify_poem(poem)
    if classification is None:
        return []

    variants = instruction_variants(classification)
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for variant in variants:
        by_type[variant["instruction_type"]].append(variant)

    selected_variants: List[Dict] = []
    for instruction_type in ("basic", "keyword", "theme", "season", "combined"):
        if by_type[instruction_type]:
            selected_variants.append(rng.choice(by_type[instruction_type]))

    remaining = [
        variant for variant in variants
        if variant["instruction"] not in {item["instruction"] for item in selected_variants}
    ]
    rng.shuffle(remaining)
    selected_variants.extend(remaining)
    selected_variants = selected_variants[:max_samples_per_poem]

    samples: List[Dict] = []
    for variant in selected_variants:
        samples.append({
            "instruction": variant["instruction"],
            "input": "",
            "output": classification.output,
            "metadata": build_metadata(classification, source_index, source_file, variant),
        })
    return samples


def read_poems(paths: List[Path]) -> Iterable[tuple[str, int, str]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                text = line.strip()
                if text:
                    yield text, index, str(path)


def select_round_robin_by_type(samples: List[Dict], total: int, rng: random.Random) -> List[Dict]:
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        groups[sample["metadata"]["poem_type"]].append(sample)
    for group in groups.values():
        rng.shuffle(group)

    selected: List[Dict] = []
    seen_outputs = set()
    poem_types = sorted(groups)
    while len(selected) < total and poem_types:
        made_progress = False
        for poem_type in list(poem_types):
            group = groups[poem_type]
            while group:
                sample = group.pop()
                output = sample["output"]
                if output in seen_outputs:
                    continue
                selected.append(sample)
                seen_outputs.add(output)
                made_progress = True
                break
            if not group:
                poem_types.remove(poem_type)
            if len(selected) >= total:
                break
        if not made_progress:
            break
    rng.shuffle(selected)
    return selected[:total]


def target_count(total: int, instruction_type: str) -> int:
    return int(total * INSTRUCTION_TYPE_TARGETS[instruction_type])


def select_from_pool(
    pool: List[Dict],
    count: int,
    rng: random.Random,
    used_pairs: set[tuple[str, str]],
) -> List[Dict]:
    available = [
        sample for sample in pool
        if (sample["instruction"], sample["output"]) not in used_pairs
    ]
    selected = select_round_robin_by_type(available, count, rng)
    for sample in selected:
        used_pairs.add((sample["instruction"], sample["output"]))
    return selected


def constrained_select(samples: List[Dict], total: int, rng: random.Random) -> List[Dict]:
    by_instruction_type: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        by_instruction_type[sample["metadata"]["instruction_type"]].append(sample)
    for group in by_instruction_type.values():
        rng.shuffle(group)

    selected: List[Dict] = []
    used_pairs: set[tuple[str, str]] = set()
    for instruction_type in INSTRUCTION_TYPE_TARGETS:
        quota = target_count(total, instruction_type)
        selected.extend(select_from_pool(by_instruction_type[instruction_type], quota, rng, used_pairs))

    if len(selected) < total:
        remaining = [
            sample for sample in samples
            if (sample["instruction"], sample["output"]) not in used_pairs
        ]
        selected.extend(select_from_pool(remaining, total - len(selected), rng, used_pairs))

    rng.shuffle(selected)
    return selected[:total]


def reservoir_add(
    groups: Dict[tuple[str, str], List[Dict]],
    seen_counts: Counter,
    key: tuple[str, str],
    sample: Dict,
    cap: int,
    rng: random.Random,
) -> None:
    seen_counts[key] += 1
    bucket = groups[key]
    if len(bucket) < cap:
        bucket.append(sample)
        return
    replacement_index = rng.randrange(seen_counts[key])
    if replacement_index < cap:
        bucket[replacement_index] = sample


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(
    input_paths: List[Path],
    output_dir: Path,
    total: int = 3000,
    seed: int = 42,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    max_samples_per_poem: int = 6,
    reservoir_cap_per_group: int = 5000,
) -> Dict:
    rng = random.Random(seed)
    grouped_candidates: Dict[tuple[str, str], List[Dict]] = defaultdict(list)
    seen_counts: Counter = Counter()
    for poem, source_index, source_file in read_poems(input_paths):
        samples = build_instruction_samples(
            poem,
            rng,
            source_index,
            source_file,
            max_samples_per_poem=max_samples_per_poem,
        )
        for sample in samples:
            key = (sample["metadata"]["instruction_type"], sample["metadata"]["poem_type"])
            reservoir_add(grouped_candidates, seen_counts, key, sample, reservoir_cap_per_group, rng)

    candidates = [
        sample
        for group in grouped_candidates.values()
        for sample in group
    ]
    selected = constrained_select(candidates, total, rng)
    if len(selected) < total:
        raise RuntimeError(f"Only built {len(selected)} samples, fewer than requested {total}.")

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    train = selected[:train_count]
    val = selected[train_count:train_count + val_count]
    test = selected[train_count + val_count:]

    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "val.jsonl", val)
    write_jsonl(output_dir / "test.jsonl", test)

    type_counts = Counter(sample["metadata"]["poem_type"] for sample in selected)
    instruction_type_counts = Counter(sample["metadata"]["instruction_type"] for sample in selected)
    theme_counts = Counter(sample["metadata"].get("constraint_theme") or sample["metadata"].get("theme") or "" for sample in selected)
    instruction_counts = Counter(sample["instruction"] for sample in selected)
    output_counts = Counter(sample["output"] for sample in selected)
    metadata = {
        "dataset_name": f"poetry-sft-v2-{total}",
        "description": "Constraint-heavy compact instruction tuning dataset generated from local processed poetry corpus.",
        "total_samples": total,
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "seed": seed,
        "max_samples_per_poem": max_samples_per_poem,
        "reservoir_cap_per_group": reservoir_cap_per_group,
        "candidate_samples": len(candidates),
        "input_files": [str(path) for path in input_paths],
        "format": {
            "instruction": "创作要求",
            "input": "保留标准字段，当前为空",
            "output": "换行后的诗词文本",
            "metadata": "体裁、标签、来源和质量约束",
        },
        "poem_type_distribution": dict(type_counts),
        "instruction_type_distribution": dict(instruction_type_counts),
        "theme_distribution": dict(theme_counts),
        "unique_instruction_count": len(instruction_counts),
        "unique_output_count": len(output_counts),
        "duplicate_output_count": sum(count - 1 for count in output_counts.values() if count > 1),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact poetry SFT samples.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["data/processed/train.txt"],
        help="Input processed poetry text files.",
    )
    parser.add_argument("--output_dir", default="data/poetry_sft_v2_10000")
    parser.add_argument("--total", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples_per_poem", type=int, default=6)
    parser.add_argument("--reservoir_cap_per_group", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_dataset(
        input_paths=[Path(path) for path in args.input],
        output_dir=Path(args.output_dir),
        total=args.total,
        seed=args.seed,
        max_samples_per_poem=args.max_samples_per_poem,
        reservoir_cap_per_group=args.reservoir_cap_per_group,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
