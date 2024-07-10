# Leave No Patient Behind: Enhancing Medication Recommendation for Rare Disease Patients

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/zzhUSTC2016/RAREMed/blob/main/LICENSE)

<div style="text-align: center;">
<img src="figs/RAREMed.png" alt="introduction" style="zoom:50%;" />
</div>


This repository provides the official PyTorch implementation and reproduction for our **SIGIR'24** paper titled **"Leave No Patient Behind: Enhancing Medication Recommendation for Rare Disease Patients"**. 

More descriptions are available via the [paper](https://arxiv.org/abs/2403.17745).
<!-- and the [slides](https://cdn.chongminggao.top/files/pdf/DORL-slides.pdf), and this Chinese [Zhihu Post](https://zhuanlan.zhihu.com/p/646690133). -->


If this work helps you, please kindly cite our papers:

```tex
@article{zhao2024leave,
  title={Leave No Patient Behind: Enhancing Medication Recommendation for Rare Disease Patients},
  author={Zhao, Zihao and Jing, Yi and Feng, Fuli and Wu, Jiancan and Gao, Chongming and He, Xiangnan},
  journal={arXiv preprint arXiv:2403.17745},
  year={2024}
}
```

## Installation

1. Clone this git repository and change directory to this repository:

   ```shell
   git clone https://github.com/zzhUSTC2016/RAREMed.git
   cd RAREMed/
   ```

2. A new [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html) is suggested. 

   ```bash
   conda create --name RAREMed
   ```

3. Activate the newly created environment.

   ```bash
   conda activate RAREMed
   ```

4. Install the required modules.

   ```bash
   sh install.sh
   ```


## Download the data
1. You must have obtained access to [MIMIC-III](https://physionet.org/content/mimiciii/) and [MIMIC-IV](https://physionet.org/content/mimiciv/) databases before running the code. 

2. Download the MIMIC-III and MIMIC-IV datasets, then unzip and put them in the `data/input/` directory. Specifically, you need to download the following files from MIMIC-III: `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, and `PROCEDURES_ICD.csv`, and the following files from MIMIC-IV: `DIAGNOSES_ICD.csv`, `PRESCRIPTIONS.csv`, and `PROCEDURES_ICD.csv`.

## Preprocess the data
Run the following command to preprocess the data:

```bash
python preprocess.py
```

If things go well, the processed data will be saved in the `data/output/` directory. You can run the models now!

## Run the models

