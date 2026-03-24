# Sparse Attention Vectors (SAVs)
---
*SAVs is an interpretable feature extraction method that enables state-of-the-art (SOTA) few-shot performance on a variety of discrete-label classification tasks: surpassing other few-shot and even LoRA-finetuned baselines.*


<p align="center">
  <img src="images/SAVs_banner.png" alt="SAVs Banner"/>
  <br>
  <a href="https://chancharikmitra.github.io/SAVs_website/">Website</a> | <a href="https://arxiv.org/abs/2412.00142">Paper</a>
</p>

### 📰 News
- [01/13/2025] 🔥 SAVs codebase released publicly!
- [04/01/2025] 🎥 Added video inference capabilities!

### Quickstart
---
More details about the components of our set up can be found later in the README.

```python
git clone https://github.com/chancharikmitra/SAVs.git
cd SAVs

conda create -n savs python=3.10 -y
conda activate savs
pip install -e .
```

To run evaluation with the new unified pipeline, prepare train/val files in `.json` or `.jsonl` with at least:

```
{"image": "image_path", "question": "Input text", "label": "desired_label", "question_id": 0}
```

Then launch experiments through Hydra from the repository root:

```bash
python main.py
```

Override model / dataset / method / evaluator from CLI, for example:

```bash
python main.py model=qwen3_vl dataset=natural_ret_sample method=zero_shot evaluator=auto
python main.py model=idefics3 dataset=sugarcrepe_sample method=sav evaluator=pair method.params.num_heads=20
```

Override custom dataset paths directly:

```bash
python main.py dataset=general_custom \
  dataset.train_path=/path/to/train.jsonl \
  dataset.val_path=/path/to/val.jsonl
```

### Method Overview
---
<p align="center">
  <img src=images/teaser.png />
</p>


**Key Problem:**
- Large Multimodal Models (LMMs) excel at generative tasks but aren't directly suited for discriminative vision-language tasks (classification, multiple-choice VQA)
- Need better ways to extract useful features from these models for discrete label predictions

**Our Solution - SAVs:**
- A finetuning-free method that extracts sparse attention head activations from LMMs
- Uses less than 1% of attention heads as features

**Key Benefits:**
- Works with any discriminative vision-language task requiring discrete labels
- Achieves SOTA performance with just few-shot examples
- Outperforms both few-shot and finetuned baselines (including LoRA)
- Scales well with additional examples
- Generalizes effectively to similar tasks
- Requires no model finetuning
- Creates robust, truly **multimodal** feature representations
  <p align="center">
    <table>
      <tr>
       <th></th>
       <th>CLIP</th>
       <th>SigLIP</th>
       <th style="border-left: 2px solid black; border-right: 2px solid black;"><b>SAVs</b></th>
      </tr>
      <tr>
       <td>Image Input</td>
       <td>✓</td>
       <td>✓</td>
       <td style="border-left: 2px solid black; border-right: 2px solid black;">✓</td>
      </tr>
      <tr>
       <td>Text Input</td>
       <td>✓</td>
       <td>✓</td>
       <td style="border-left: 2px solid black; border-right: 2px solid black;">✓</td>
      </tr>
      <tr>
       <td>Interleaved Input</td>
       <td>✗</td>
       <td>✗</td>
       <td style="border-left: 2px solid black; border-right: 2px solid black;">✓</td>
      </tr>
    </table>
  </p>

For more information, please refer to our [paper](https://arxiv.org/abs/2412.00142)!

### 💻 Setup
---
To get started, first clone our repo and set up the environment:

```bash
git clone https://github.com/chancharikmitra/SAVs.git
cd SAVs

conda create -n savs python=3.10 -y
conda activate savs
pip install -e .
```

#### Running SAVs / Zero-shot

The new codebase unifies both `zero_shot` and `sav` under one Hydra entry (`main.py`).

- Methods: `zero_shot`, `sav`
- Models (current first-class support): `qwen3_vl`, `idefics3`
- Evaluators: `raw`, `pair`, `naturalbench_group`, or `auto` (from dataset config)

SAV run example:

```bash
python main.py model=qwen3_vl dataset=pets_sample method=sav evaluator=raw
```

Zero-shot run example:

```bash
python main.py model=idefics3 dataset=naturalbench_ret method=zero_shot evaluator=naturalbench_group
```

Metrics and prediction files are saved to `run.output_dir` (default: `outputs/eval`).

#### Models
Current Hydra pipeline supports `qwen3_vl` and `idefics3` out of the box. Add new models by extending `src/models` and registering them in `src/models/__init__.py`.

#### Datasets
Our method can be applied to any discriminative, discrete-label VL task. We provide a variety of examples on how to format datasets (found in the `data` folder). Adding a new dataset is simple:

1. Format a training and validation set with keys `image`, `question`, `label` (optional `question_id`).
2. Point `dataset.train_path` and `dataset.val_path` to your files via Hydra CLI override or a new config under `conf/dataset`.

You may choose to change the default number of examples and heads used. But we find 20 examples and 20 heads is enough to yield state-of-the-art performance on a variety of VL tasks: **outperforming even LoRA at this sample complexity!**

Note regarding evaluation: use `raw` for standard classification, `pair` for pair-based benchmarks like SUGARCREPE, and `naturalbench_group` for grouped NaturalBench metrics (Q-Acc / I-Acc / G-Acc / Raw) on the `naturalbench_ret` dataset.

Other Notes:
 - NaturalBench Images can be downloaded at the following [link](https://huggingface.co/datasets/BaiqiL/naturalbench_pictures/blob/main/raw_images.zip)

#### Dataset Extraction & testbed Integration

**Migration from external location:** If you have archives in another directory (e.g. `/root/autodl-tmp`), copy them into `data/` first:

```bash
cp /path/to/102flowers.tgz annotations.tar.gz images.tar.gz cub200.zip EuroSAT_MS.zip tiny-imagenet-200.zip val.zip val2017.zip /path/to/FGVC/data/
```

Then extract all archives:

```bash
bash scripts/extract_datasets.sh
```

This supports `.zip`, `.tgz`, and `.tar.gz`. Data layout under `data/`:

| Archive | Output | Dataset |
|---------|--------|---------|
| BaiqiL___natural_bench.zip | natural_bench/ | NaturalBench |
| BLINK-Benchmark___blink.zip | blink/ | BLINK |
| openkg___m_halu_bench.zip | m_halu_bench/ | M-Halu |
| ys-zong___vl_guard.zip | vl_guard/ | VL-Guard |
| 102flowers.tgz | flowers102/ | Oxford Flowers 102 |
| annotations.tar.gz | pets/annotations/ | Oxford-IIIT Pet |
| images.tar.gz | pets/images/ | Oxford-IIIT Pet |
| cub200.zip | cub200/ | CUB-200-2011 |
| EuroSAT_MS.zip | eurosat/ | EuroSAT (multi-spectral) |
| tiny-imagenet-200.zip | tiny_imagenet/ | Tiny-ImageNet |
| val2017.zip | coco/val2017/ | COCO val2017 |
| val.zip | coco/val/ | COCO val |

COCO val2017 is used by SUGARCREPE and NaturalBench-Retrieval (image-text matching). Other image classification datasets (Flowers, Pet, CUB, EuroSAT, Tiny-ImageNet) are for fine-grained classification.

**Path configuration:** Edit `paths.py` or set environment variables:
- `FGVC_DATA_ROOT` – root for datasets (default: `data/`)
- `FGVC_DATASET_ROOT` – root for annotation JSON/JSONL (default: `dataset/`)

Annotations in `dataset/` may use absolute paths from other machines. Path remapping in `testbed/paths.py` rewrites them to local `data/` paths.

**Adding a new dataset to testbed:**
1. Create `testbed/data/<name>/` with `<name>.py` (Builder) and `__init__.py` (retriever + postprocess).
2. Use `testbed.data.common.split_generators` and `@register_dataset_retriever` / `@register_postprocess`.
3. Add path remapping in `testbed/paths.py` if annotation paths differ from your layout.

Verify annotations and path remapping:

```bash
uv run python scripts/verify_datasets.py
```

Check dataset availability (annotations, files, missing data):

```bash
uv run python scripts/check_datasets.py
```

**Data format note:** NaturalBench and VL-Guard from HuggingFace use Arrow format (images in .arrow files). The `dataset/` annotations expect loose image paths. For full compatibility, either use `datasets.load_dataset` with the Arrow data, or obtain the raw image layouts (NaturalBench-Retrieval.zip, VLGuard train.zip/test.zip) and place images accordingly.

### 📝 Citation
---
If you found our work useful, please consider starring and citing. Thank you!
```latex
@article{mitra2024sparse,
  title={Sparse Attention Vectors: Generative Multimodal Model Features Are Discriminative Vision-Language Classifiers},
  author={Mitra, Chancharik and Huang, Brandon and Chai, Tianning and Lin, Zhiqiu and Arbelle, Assaf and Feris, Rogerio and Karlinsky, Leonid and Darrell, Trevor and Ramanan, Deva and Herzig, Roei},
  journal={arXiv preprint arXiv:2412.00142},
  year={2024}
}
```
