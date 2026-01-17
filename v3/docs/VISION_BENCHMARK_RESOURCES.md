# Vision Benchmark Image Resources

**Purpose**: Real images for L2 (Vision) tool-gated tasks

---

## 1. MMMU (Massive Multi-discipline Multimodal Understanding)

**Description**: 11.5K college-level problems requiring vision across 30 subjects

**Source**: 
- Hugging Face: `MMMU/MMMU`
- Paper: ICLR 2024

**Download**:
```python
from datasets import load_dataset

# Load MMMU dataset (includes images)
dataset = load_dataset("MMMU/MMMU", split="validation")

# Access images
for example in dataset:
    image = example['image']  # PIL Image
    question = example['question']
    answer = example['answer']
```

---

## 2. MathVista (Mathematical Visual Understanding)

**Description**: 6K math problems requiring visual reasoning (charts, diagrams, geometry)

**Source**:
- Hugging Face: `AI4Math/MathVista`
- Paper: ICLR 2024

**Download**:
```python
from datasets import load_dataset

dataset = load_dataset("AI4Math/MathVista", split="testmini")
```

---

## 3. ChartQA (Chart Question Answering)

**Description**: Questions about bar charts, line graphs, pie charts

**Source**:
- Hugging Face: `ahmed-masry/ChartQA`
- GitHub: https://github.com/vis-nlp/ChartQA

**Download**:
```python
from datasets import load_dataset

dataset = load_dataset("ahmed-masry/ChartQA", split="test")
```

---

## 4. DocVQA (Document Visual QA)

**Description**: Questions about scanned documents

**Source**:
- Hugging Face: `lmms-lab/DocVQA`
- Official: https://www.docvqa.org/

---

## 5. ScienceQA (Science Visual QA)

**Description**: Science questions with diagrams and images

**Source**:
- Hugging Face: `derek-thomas/ScienceQA`
- GitHub: https://github.com/lupantech/ScienceQA

---

## 6. VQAv2 (Visual Question Answering v2)

**Description**: General visual question answering

**Source**:
- Hugging Face: `HuggingFaceM4/VQAv2`
- Official: https://visualqa.org/

---

## Quick Start Script

```python
"""Download sample images for V3 vision tasks."""
from datasets import load_dataset
import os

os.makedirs('data/images', exist_ok=True)

# Download ChartQA samples (best for tool-gated tasks)
print("Downloading ChartQA samples...")
charts = load_dataset("ahmed-masry/ChartQA", split="test[:50]")
for i, ex in enumerate(charts):
    if ex['image']:
        ex['image'].save(f'data/images/chart_{i}.png')
        
print(f"Downloaded {len(charts)} chart images")
```

---

## How Others Have Done It

### LLaVA
- GitHub: https://github.com/haotian-liu/LLaVA
- Uses HuggingFace datasets for images

### Qwen-VL
- GitHub: https://github.com/QwenLM/Qwen-VL
- Multi-benchmark evaluation

### CogVLM
- GitHub: https://github.com/THUDM/CogVLM
- Provides download scripts

---

## Recommended for V3

| Benchmark | Images | Why Tool-Gated |
|-----------|--------|----------------|
| ChartQA | 21K | Must see chart to answer |
| MathVista | 6K | Must see diagram to solve |
| DocVQA | 50K | Must read document |
| MMMU | 11.5K | College-level vision |

---

*Created: 2026-01-15*
