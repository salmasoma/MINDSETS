# 🧠 MINDSETS - Dementia Differential Diagnosis

## 📌 Overview

This project processes MRI scans for segmentation, radiomics feature extraction, and classification into four classes: **Control, Mild Cognitive Impairment (MCI), Alzheimer's Disease (AD), and Vascular Dementia (VaD)**. The pipeline integrates **SynthSeg segmentation** and a trained classification model.

## 🏗️ Architecture

![Model Architecture](./Figures/MINDSETS_Arch.png)

The pipeline consists of three main steps:

1. 🧠 **Segmentation**: MRI scans are segmented using the SynthSeg model.
2. **📊 Feature Extraction**: Radiomics features are extracted from segmented brain structures.
3. **🤖 Classification**: Extracted features are fed into a pre-trained classifier to predict the disease category.

## 🛠 Installation

Ensure you have the required dependencies installed:

`pip install -r requirements.txt`

## 🚀 Usage

To run the MRI classification pipeline, use the following command:

### Basic Usage:

`python scripts/inference.py --input <path_to_mri.nii.gz> --output <output_directory> --model <model_path>`

### With verbose output:

`python scripts/inference.py --input <path_to_mri.nii.gz> --output <output_directory> --model <model_path> --verbose`

## 📂 Output Files

After running the pipeline, the following output files will be generated:

* 🏷 **Segmentation Mask**: `<span><filename>_synthseg.nii.gz</span>`
* 📜 **Extracted Features**: `<span><filename>_radiomics.csv</span>`
* 📑 **Classification Results**: `<span><filename>_classification.txt</span>`

## 📝 Citation

If you use this project in your research, please cite:

> @article{hassan2024mindsets,
> title={MINDSETS: Multi-omics Integration with Neuroimaging for Dementia Subtyping and Effective Temporal Study},
> author={Hassan, Salma and Akaila, Dawlat and Arjemandi, Maryam and Papineni, Vijay and Yaqub, Mohammad},
> journal={arXiv preprint arXiv:2411.04155},
> year={2024}
> }
