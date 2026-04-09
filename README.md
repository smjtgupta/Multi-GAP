# Multi-GAP

Official implementation of ["Fairness-Aware Multi-Group Target Detection in Online Discussion"](). Accepted at [_ACM Conference on Fairness, Accountability, and Transparency (FAccT 2026)_](https://facctconference.org/2026/index.html).

---

## Install

```bash
git clone https://github.com/smjtgupta/Multi-GAP
cd Multi-GAP
pip install -r requirements.txt
```

## Run Experiments

To reproduce experimental results simply run the provided notebook. <br>
It uses the Measuring Hate Speech (MHS) corpus [dataset](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech). <br>

If you want to use the **Multi-GAP** loss function directly in your application, please refer to the file _multi_gap.py_, containing the vectorized version of the function to allow scaling to multiple groups.

## Citation

If you use **Multi-GAP** in your own work, please cite the following paper:

```bib

```
