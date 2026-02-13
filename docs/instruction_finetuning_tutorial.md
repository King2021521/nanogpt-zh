# 诗词创作指令微调训练数据准备教程

> @Author xiaomin.zhang

## 1. 概述

本教程将指导你准备用于诗词创作的指令微调训练数据。通过这份数据，模型将学会根据用户的提示词（如主题、风格、情感等）创作符合要求的诗词。

## 2. 数据规模建议

### 2.1 最小规模
- **最小训练集**: 1,000 - 2,000 条高质量样本
- **验证集**: 200 - 300 条
- **测试集**: 200 - 300 条

### 2.2 推荐规模
- **训练集**: 5,000 - 10,000 条
- **验证集**: 500 - 1,000 条
- **测试集**: 500 - 1,000 条

### 2.3 理想规模
- **训练集**: 20,000+ 条
- **验证集**: 2,000+ 条
- **测试集**: 2,000+ 条

## 3. 数据结构设计

### 3.1 JSON格式（推荐）

每条数据包含以下字段：

```json
{
  "instruction": "用户的指令/提示词",
  "input": "可选的额外输入信息",
  "output": "模型应该生成的诗词",
  "metadata": {
    "poem_type": "诗词类型",
    "style": "风格",
    "dynasty": "朝代",
    "difficulty": "难度等级"
  }
}
```

### 3.2 JSONL格式（流式处理友好）

每行一个JSON对象：

```jsonl
{"instruction": "...", "input": "...", "output": "...", "metadata": {...}}
{"instruction": "...", "input": "...", "output": "...", "metadata": {...}}
```

### 3.3 CSV格式（简单场景）

```csv
instruction,input,output,poem_type,style
"写一首关于春天的五言绝句","","春风拂柳绿，...",五言绝句,清新
```

## 4. 详细数据示例

### 4.1 基础示例（五言绝句）

```json
{
  "instruction": "请写一首关于春天的五言绝句",
  "input": "",
  "output": "春风拂柳绿，\n燕子绕梁飞。\n桃花三月雨，\n诗意满园归。",
  "metadata": {
    "poem_type": "五言绝句",
    "style": "清新自然",
    "dynasty": "现代",
    "difficulty": "初级"
  }
}
```

### 4.2 带情感要求的示例

```json
{
  "instruction": "创作一首表达思乡之情的七言绝句",
  "input": "",
  "output": "明月千里照故乡，\n秋风落叶满庭霜。\n游子天涯心未老，\n梦回故里泪两行。",
  "metadata": {
    "poem_type": "七言绝句",
    "style": "婉约",
    "emotion": "思乡",
    "difficulty": "中级"
  }
}
```

### 4.3 带主题和风格的示例

```json
{
  "instruction": "以'月'为主题，写一首豪放派风格的七言律诗",
  "input": "",
  "output": "长空万里月如钩，\n照破山河几度秋。\n壮士胸中藏日月，\n英雄剑下斩千愁。\n江山如画风云起，\n天地无边志气酬。\n举杯邀月共此夜，\n笑看人间任自由。",
  "metadata": {
    "poem_type": "七言律诗",
    "style": "豪放",
    "theme": "月",
    "difficulty": "高级"
  }
}
```

### 4.4 带约束条件的示例

```json
{
  "instruction": "写一首五言绝句，要求包含'风'、'雨'两个字，表达离别之情",
  "input": "关键词：风、雨",
  "output": "风起雨纷纷，\n离人泪满襟。\n天涯各一方，\n何日再相寻。",
  "metadata": {
    "poem_type": "五言绝句",
    "style": "婉约",
    "emotion": "离别",
    "keywords": ["风", "雨"],
    "difficulty": "中级"
  }
}
```

### 4.5 模仿古人风格的示例

```json
{
  "instruction": "模仿李白的风格，创作一首关于饮酒的诗",
  "input": "风格：李白",
  "output": "青天有月来几时，\n我今停杯一问之。\n人生得意须尽欢，\n莫使金樽空对月。\n天生我材必有用，\n千金散尽还复来。\n烹羊宰牛且为乐，\n会须一饮三百杯。",
  "metadata": {
    "poem_type": "古体诗",
    "style": "豪放",
    "imitate": "李白",
    "theme": "饮酒",
    "difficulty": "高级"
  }
}
```

### 4.6 现代主题的示例

```json
{
  "instruction": "用古诗词的形式描写现代都市生活",
  "input": "主题：现代都市",
  "output": "高楼林立入云霄，\n车水马龙夜未消。\n霓虹闪烁千万色，\n都市繁华竞折腰。",
  "metadata": {
    "poem_type": "七言绝句",
    "style": "现代",
    "theme": "都市",
    "difficulty": "中级"
  }
}
```

## 5. 完整数据集示例文件

### 5.1 训练数据示例（train.jsonl）

```jsonl
{"instruction": "写一首描写夏天的五言绝句", "input": "", "output": "荷花映日红，\n蝉鸣柳树中。\n清风送凉意，\n夏日亦从容。", "metadata": {"poem_type": "五言绝句", "style": "清新", "difficulty": "初级"}}
{"instruction": "创作一首表达友情的七言绝句", "input": "", "output": "相逢一笑泯恩仇，\n把酒言欢话旧游。\n天涯若比邻在望，\n知己难求世上留。", "metadata": {"poem_type": "七言绝句", "style": "豪放", "emotion": "友情", "difficulty": "中级"}}
{"instruction": "以'山'为题，写一首五言律诗", "input": "", "output": "青山横北郭，\n白水绕东城。\n此地一为别，\n孤蓬万里征。\n浮云游子意，\n落日故人情。\n挥手自兹去，\n萧萧班马鸣。", "metadata": {"poem_type": "五言律诗", "style": "古典", "theme": "山", "difficulty": "高级"}}
{"instruction": "写一首关于秋天的词，风格婉约", "input": "", "output": "秋风起兮白云飞，\n草木黄落兮雁南归。\n兰有秀兮菊有芳，\n怀佳人兮不能忘。", "metadata": {"poem_type": "古体诗", "style": "婉约", "season": "秋", "difficulty": "中级"}}
{"instruction": "创作一首赞美梅花的七言绝句", "input": "", "output": "墙角数枝梅独立，\n凌寒独自开花时。\n遥知不是雪纷飞，\n为有暗香来袭之。", "metadata": {"poem_type": "七言绝句", "style": "清新", "theme": "梅花", "difficulty": "中级"}}
```

### 5.2 验证数据示例（val.jsonl）

```jsonl
{"instruction": "写一首关于冬天的五言绝句", "input": "", "output": "雪落梅花白，\n寒风透骨来。\n炉火红似锦，\n温酒待君开。", "metadata": {"poem_type": "五言绝句", "style": "清新", "difficulty": "初级"}}
{"instruction": "创作一首表达爱情的七言绝句", "input": "", "output": "红豆生南国春来，\n愿君多采撷几回。\n此物最相思难忘，\n一枝一叶总关怀。", "metadata": {"poem_type": "七言绝句", "style": "婉约", "emotion": "爱情", "difficulty": "中级"}}
```

## 6. 指令类型多样性

为了让模型具有更好的泛化能力，指令应该包含多种类型：

### 6.1 直接创作类
- "写一首关于XX的诗"
- "创作一首XX风格的诗词"
- "作一首XX言XX句"

### 6.2 带约束条件类
- "写一首包含'XX'字的诗"
- "创作一首押XX韵的诗"
- "写一首每句都有'XX'字的诗"

### 6.3 情感表达类
- "表达XX情感的诗"
- "描写XX心情的诗词"
- "抒发XX之情的作品"

### 6.4 模仿风格类
- "模仿XX诗人的风格"
- "学习XX朝代的诗风"
- "借鉴XX流派的特点"

### 6.5 场景描写类
- "描写XX场景"
- "刻画XX画面"
- "展现XX意境"

### 6.6 主题创作类
- "以XX为主题"
- "围绕XX展开"
- "聚焦XX话题"

## 7. 数据质量要求

### 7.1 诗词质量
- ✅ 符合格律要求（平仄、押韵）
- ✅ 意境优美，语言流畅
- ✅ 内容与指令高度相关
- ✅ 避免生硬、牵强的表达

### 7.2 指令质量
- ✅ 清晰明确，不产生歧义
- ✅ 覆盖多种难度级别
- ✅ 包含丰富的变化形式
- ✅ 贴近真实用户需求

### 7.3 数据平衡性
- ✅ 各种诗词类型均衡分布
- ✅ 不同难度级别合理配比
- ✅ 多样化的主题和风格
- ✅ 避免重复和雷同

## 8. 数据生成方法

### 8.1 人工创作（推荐）
- 由专业人员根据指令创作
- 质量最高，但成本较大
- 适合核心数据集

### 8.2 古诗词改编
- 基于经典古诗词设计指令
- 质量有保障，效率较高
- 需注意版权问题

### 8.3 AI辅助生成
- 使用大模型生成初稿
- 人工审核和修改
- 效率高，需要质量把控

### 8.4 众包标注
- 通过平台招募标注人员
- 适合大规模数据生成
- 需要严格的质量控制流程

## 9. 数据文件组织

```
data/
├── poetry_instruction/
│   ├── train.jsonl          # 训练集
│   ├── val.jsonl            # 验证集
│   ├── test.jsonl           # 测试集
│   ├── metadata.json        # 数据集元信息
│   └── README.md            # 数据说明文档
├── raw/                     # 原始数据
│   ├── classical_poems/     # 古典诗词
│   └── modern_poems/        # 现代诗词
└── processed/               # 处理后的数据
    └── tokenized/           # 分词后的数据
```

## 10. 数据预处理脚本示例

创建 `prepare_data.py`:

```python
# @Author xiaomin.zhang

import json
import random
from pathlib import Path
from typing import List, Dict

def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL格式的数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """保存为JSONL格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def format_instruction_data(instruction: str, input_text: str, output: str, metadata: Dict = None) -> Dict:
    """格式化单条指令数据"""
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "metadata": metadata or {}
    }

def split_dataset(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    """划分数据集"""
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        'train': data[:train_end],
        'val': data[train_end:val_end],
        'test': data[val_end:]
    }

def validate_poem_format(poem: str, poem_type: str) -> bool:
    """验证诗词格式"""
    lines = poem.strip().split('\n')
    
    if poem_type == "五言绝句":
        return len(lines) == 4 and all(len(line.replace('，', '').replace('。', '')) == 5 for line in lines)
    elif poem_type == "七言绝句":
        return len(lines) == 4 and all(len(line.replace('，', '').replace('。', '')) == 7 for line in lines)
    elif poem_type == "五言律诗":
        return len(lines) == 8 and all(len(line.replace('，', '').replace('。', '')) == 5 for line in lines)
    elif poem_type == "七言律诗":
        return len(lines) == 8 and all(len(line.replace('，', '').replace('。', '')) == 7 for line in lines)
    
    return True  # 其他类型暂不验证

def create_sample_dataset():
    """创建示例数据集"""
    samples = [
        {
            "instruction": "写一首关于春天的五言绝句",
            "input": "",
            "output": "春风拂柳绿，\n燕子绕梁飞。\n桃花三月雨，\n诗意满园归。",
            "metadata": {"poem_type": "五言绝句", "style": "清新自然", "difficulty": "初级"}
        },
        {
            "instruction": "创作一首表达思乡之情的七言绝句",
            "input": "",
            "output": "明月千里照故乡，\n秋风落叶满庭霜。\n游子天涯心未老，\n梦回故里泪两行。",
            "metadata": {"poem_type": "七言绝句", "style": "婉约", "emotion": "思乡", "difficulty": "中级"}
        },
        # 添加更多样本...
    ]
    
    return samples

def main():
    """主函数"""
    # 创建输出目录
    output_dir = Path("data/poetry_instruction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成示例数据
    data = create_sample_dataset()
    
    # 划分数据集
    splits = split_dataset(data)
    
    # 保存数据
    save_jsonl(splits['train'], output_dir / "train.jsonl")
    save_jsonl(splits['val'], output_dir / "val.jsonl")
    save_jsonl(splits['test'], output_dir / "test.jsonl")
    
    # 保存元信息
    metadata = {
        "total_samples": len(data),
        "train_samples": len(splits['train']),
        "val_samples": len(splits['val']),
        "test_samples": len(splits['test']),
        "poem_types": ["五言绝句", "七言绝句", "五言律诗", "七言律诗"],
        "styles": ["清新", "婉约", "豪放", "古典"],
        "difficulty_levels": ["初级", "中级", "高级"]
    }
    
    with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"数据集准备完成！")
    print(f"训练集: {len(splits['train'])} 条")
    print(f"验证集: {len(splits['val'])} 条")
    print(f"测试集: {len(splits['test'])} 条")

if __name__ == "__main__":
    main()
```

## 11. 数据增强技巧

### 11.1 指令改写
将同一个指令用不同方式表达：
- "写一首关于春天的诗" 
- "创作一首描写春天的诗词"
- "作一首春天主题的诗"

### 11.2 多样本生成
对同一指令生成多个不同的诗词输出

### 11.3 难度递进
从简单到复杂逐步增加约束条件

### 11.4 风格迁移
同一主题用不同风格表达

## 12. 评估指标

### 12.1 自动评估
- **格律准确率**: 是否符合诗词格律
- **BLEU/ROUGE**: 与参考诗词的相似度
- **困惑度**: 模型生成的流畅度

### 12.2 人工评估
- **相关性**: 是否符合指令要求 (1-5分)
- **创意性**: 是否有新意 (1-5分)
- **艺术性**: 意境和美感 (1-5分)
- **流畅度**: 语言是否通顺 (1-5分)

## 13. 常见问题

### Q1: 数据量不够怎么办？
A: 可以从以下途径扩充：
- 使用公开的古诗词数据库
- AI辅助生成后人工筛选
- 数据增强技术
- 众包标注

### Q2: 如何保证诗词质量？
A: 建议：
- 建立严格的审核标准
- 多人交叉验证
- 使用专业人员审核
- 建立质量评分体系

### Q3: 如何处理版权问题？
A: 注意：
- 古代诗词（作者去世70年以上）无版权限制
- 现代诗词需获得授权
- 自行创作的内容无版权问题
- 标注数据来源

## 14. 下一步

准备好数据后，你可以：

1. 使用准备好的数据进行模型微调
2. 评估模型在测试集上的表现
3. 根据评估结果迭代优化数据
4. 部署模型并收集用户反馈
5. 持续改进数据和模型

## 15. 参考资源

- 古诗词数据库: [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)
- 指令微调框架: Stanford Alpaca, LLaMA-Factory
- 评估工具: NLTK, jieba, pypinyin

---

**最后更新**: 2026-02-13
**作者**: xiaomin.zhang
