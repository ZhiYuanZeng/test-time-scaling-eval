---
license: apache-2.0
language:
- en
tags:
- math
- olympiads
size_categories:
- 1K<n<10K
---


![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/65ae21adabf6d1ccb795e9a4/2K48kJlYndyPbiwVqwaRj.jpeg)

# Dataset Card for Omni-MATH

<!-- Provide a quick summary of the dataset. -->

Recent advancements in AI, particularly in large language models (LLMs), have led to significant breakthroughs in mathematical reasoning capabilities. However, existing benchmarks like GSM8K or MATH are now being solved with high accuracy (e.g., OpenAI o1 achieves 94.8% on MATH dataset), indicating their inadequacy for truly challenging these models. To mitigate this limitation, we propose a comprehensive and challenging benchmark specifically designed to assess LLMs' mathematical reasoning at the Olympiad level. Unlike existing Olympiad-related benchmarks, our dataset focuses exclusively on mathematics and comprises a vast collection of 4428 competition-level problems. These problems are meticulously categorized into 33 (and potentially more) sub-domains and span across 10 distinct difficulty levels, enabling a nuanced analysis of model performance across various mathematical disciplines and levels of complexity.

* Project Page: https://omni-math.github.io/
* Github Repo: https://github.com/KbsdJames/Omni-MATH
* Omni-Judge (opensource evaluator of this dataset): https://huggingface.co/KbsdJames/Omni-Judge

## Dataset Details


## Uses

<!-- Address questions around how the dataset is intended to be used. -->
```python
from datasets import load_dataset
dataset = load_dataset("KbsdJames/Omni-MATH")

```
For further examination of the model, please refer to our github repository: https://github.com/KbsdJames/Omni-MATH

## Citation
If you find our code and dataset helpful, welcome to cite our paper.
```
@misc{gao2024omnimathuniversalolympiadlevel,
      title={Omni-MATH: A Universal Olympiad Level Mathematic Benchmark For Large Language Models}, 
      author={Bofei Gao and Feifan Song and Zhe Yang and Zefan Cai and Yibo Miao and Qingxiu Dong and Lei Li and Chenghao Ma and Liang Chen and Runxin Xu and Zhengyang Tang and Benyou Wang and Daoguang Zan and Shanghaoran Quan and Ge Zhang and Lei Sha and Yichang Zhang and Xuancheng Ren and Tianyu Liu and Baobao Chang},
      year={2024},
      eprint={2410.07985},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.07985}, 
}
```