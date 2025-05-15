# 🧠 MINDSETS - Dementia Differential Diagnosis

## 📌 Overview

This project processes MRI scans for segmentation, radiomics feature extraction, and classification into four classes: **Control, Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD), and Vascular Dementia (VaD)**. The pipeline integrates **SynthSeg segmentation** and a trained classification model.

## 🏗️ Architecture

![Model Architecture](Figures/MINDSETS_Arch.png)

The pipeline consists of three main steps:

1. 🧠 **Segmentation**: MRI scans are segmented using the SynthSeg model.
2. **📊 Feature Extraction**: Radiomics features are extracted from segmented brain structures.
3. **🤖 Classification**: Extracted features are fed into a pre-trained classifier to predict the disease category.

## 🛠 Installation

Ensure you have the required dependencies installed:

```python
#Create Environment
conda create -n MINDSETS python=3.8
#Activate Environment
conda activate MINDSETS
#Clone Repo
git clone https://github.com/salmasoma/MINDSETS/
cd MINDSETS
#Install requirements
pip install -r requirements.txt
```

**Download model weights:** [Here]([MINDSETS_MRI_Multiclass.pt](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/salma_hassan_mbzuai_ac_ae/ERJ2Tccn3JFBq3AA-Qeydy8B0MVu3xzEKf5kzRZ97faDPg?e=rgBoe9)https://)

## 🚀 Usage

To run the MRI classification pipeline, use the following command:

### Basic Usage:

```python
python scripts/inference.py --input <path_to_mri.nii.gz> --output <output_directory> --model <model_path>
```

### With verbose output:

```python
python scripts/inference.py --input <path_to_mri.nii.gz> --output <output_directory> --model <model_path> --verbose
```

## 📂 Output Files

After running the pipeline, the following output files will be generated:

* 🏷 **Segmentation Mask**: `<span><filename>_synthseg.nii.gz</span>`
* 📜 **Extracted Features**: `<span><filename>_radiomics.csv</span>`
* 📑 **Classification Results**: `<span><filename>_classification.txt and <filename>_classification.json</span>`

## 📥 Demo & Paper

**Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/SalmaHassan/MINDSETS-APP)

**Paper:** [Link](https://www.nature.com/articles/s41598-025-97674-0https://)

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@article{hassan2025mindsets,
  title={MINDSETS: Multi-omics Integration with Neuroimaging for Dementia Subtyping and Effective Temporal Study},
  author={Hassan, Salma and Akaila, Dawlat and Arjemandi, Maryam and Papineni, Vijay and Yaqub, Mohammad},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={1--12},
  year={2025},
  publisher={Nature Publishing Group}
}
```
