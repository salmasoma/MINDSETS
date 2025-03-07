import argparse
import nibabel as nib
import os
import tempfile
from predict_synthseg import predict
import time
from radiomics import featureextractor
import SimpleITK as sitk
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import shutil

def extract_features(label, structure, extractor, mri_image, mask_image):
    """
    Extract radiomics features for a specific label and structure
    
    Args:
        label (int): Label value for the structure
        structure (str): Name of the brain structure
        extractor: PyRadiomics feature extractor
        mri_image: SimpleITK image of the MRI
        mask_image: SimpleITK image of the segmentation mask
    
    Returns:
        dict: Dictionary of extracted features with structure name added
    """
    # Create a binary mask for this specific label
    label_mask = sitk.GetImageFromArray((sitk.GetArrayFromImage(mask_image) == label).astype(int))
    label_mask.CopyInformation(mask_image)
    
    # Check if the mask contains the label
    if sitk.GetArrayFromImage(label_mask).sum() == 0:
        raise ValueError(f"Label {label} not found in segmentation mask")
    
    # Extract features
    features = extractor.execute(mri_image, label_mask, label=int(label))
    features['Structure'] = structure
    return features

def process_mri(input_file, output_dir, model_path=None, verbose=False, clean=True):
    """
    Process an MRI scan through segmentation, feature extraction and classification
    
    Args:
        input_file (str): Path to input MRI file (.nii or .nii.gz)
        output_dir (str): Directory to store outputs
        model_path (str): Path to the classification model (.pkl)
        verbose (bool): Whether to print detailed progress
        clean (bool): Whether to clean temporary files after processing
    
    Returns:
        dict: Classification results with class probabilities
    """
    start_time = time.time()
    
    # Create a temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    
    # Define labels and structures
    labels_structures = {
        2: "left cerebral white matter",
        3: "left cerebral cortex",
        4: "left lateral ventricle",
        5: "left inferior lateral ventricle",
        7: "left cerebellum white matter",
        8: "left cerebellum cortex",
        10: "left thalamus",
        11: "left caudate",
        12: "left putamen",
        13: "left pallidum",
        14: "3rd ventricle",
        15: "4th ventricle",
        16: "brain-stem",
        17: "left hippocampus",
        18: "left amygdala",
        26: "left accumbens area",
        24: "CSF",
        28: "left ventral DC",
        41: "right cerebral white matter",
        42: "right cerebral cortex",
        43: "right lateral ventricle",
        44: "right inferior lateral ventricle",
        46: "right cerebellum white matter",
        47: "right cerebellum cortex",
        49: "right thalamus",
        50: "right caudate",
        51: "right putamen",
        52: "right pallidum",
        53: "right hippocampus",
        54: "right amygdala",
        58: "right accumbens area",
        60: "right ventral DC"
    }
    
    # Load the classification model
    if model_path is None:
        model_path = "../models/model_dfg.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Get filename info
    filename = os.path.basename(input_file)
    filename_no_ext = Path(filename).stem
    if filename_no_ext.endswith('.nii'):
        filename_no_ext = filename_no_ext[:-4]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if verbose:
        print(f"[Step 1/3] Running Segmentation model...")
    
    # Run segmentation directly from source to temporary directory
    volumes_file = os.path.join(output_dir, f"{filename_no_ext}_volumes.csv")
    predict(path_images=input_file,
            path_segmentations=output_dir,
            path_model_segmentation="../models/synthseg_2.0.h5",
            labels_segmentation="../data/labels_classes_priors/synthseg_segmentation_labels_2.0.npy",
            robust=False,
            fast=True,
            v1=False,
            do_parcellation=False,
            n_neutral_labels=19,
            names_segmentation="../data/labels_classes_priors/synthseg_segmentation_names_2.0.npy",
            labels_denoiser="../data/labels_classes_priors/synthseg_denoiser_labels_2.0.npy",
            path_posteriors=None,
            path_resampled=None,
            path_volumes=volumes_file,
            path_model_parcellation="../models/synthseg_parc_2.0.h5",
            labels_parcellation="../data/labels_classes_priors/synthseg_parcellation_labels.npy",
            names_parcellation="../data/labels_classes_priors/synthseg_parcellation_names.npy",
            path_model_qc="../models/synthseg_qc_2.0.h5",
            labels_qc="../data/labels_classes_priors/synthseg_qc_labels_2.0.npy",
            path_qc_scores=None,
            names_qc="../data/labels_classes_priors/synthseg_qc_names_2.0.npy",
            cropping=None,
            topology_classes="../data/labels_classes_priors/synthseg_topological_classes_2.0.npy",
            ct=False)
    
    if verbose:
        print(f"[Step 2/3] Extracting Radiomics...")
    
    # Find the segmentation file
    mask_file = os.path.join(output_dir, f"{filename_no_ext}_synthseg.nii.gz")
    
    # Extract radiomics features
    if verbose:
        print("Initializing radiomics feature extractor...")
    
    extractor = featureextractor.RadiomicsFeatureExtractor()
    
    # Load images
    mri_image = sitk.ReadImage(input_file)
    mask_image = sitk.ReadImage(mask_file)
    
    # Resample mask to match the size of the MRI image
    if verbose:
        print("Resampling mask to match MRI dimensions...")
    
    mask_image = sitk.Resample(mask_image, mri_image, sitk.Transform(), 
                               sitk.sitkNearestNeighbor, 0.0, mask_image.GetPixelID())
    
    # Extract features for each brain structure
    results_list = []
    if verbose:
        print(f"Extracting features for {len(labels_structures)} brain structures...")
    
    for label, structure in labels_structures.items():
        if verbose:
            print(f"  Processing {structure} (label {label})...")
        
        try:
            features = extract_features(label, structure, extractor, mri_image, mask_image)
            results_list.append(features)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to extract features for {structure} (label {label}): {str(e)}")
    
    # Convert to DataFrame
    if verbose:
        print("Processing extracted features...")
    
    results_df = pd.DataFrame(results_list)
    
    # Drop diagnostic columns which are not needed for classification
    results_df = results_df[results_df.columns.drop(list(results_df.filter(regex='diagnostics')))]
    
    # Reorder the structure column to the start
    if 'Structure' in results_df.columns:
        results_df = results_df[["Structure"] + [col for col in results_df.columns if col != "Structure"]]
    
    # Save extracted features for reference
    features_file = os.path.join(output_dir, f"{filename_no_ext}_radiomics.csv")
    results_df.to_csv(features_file, index=False)
    
    if verbose:
        print(f"Extracted {len(results_df)} feature sets across {len(results_list)} structures")
        print(f"Features saved to {features_file}")
    
    # Remove 'Structure' column for prediction
    if 'Structure' in results_df.columns:
        results_df = results_df.drop(columns=['Structure'])
    
    if verbose:
        print(f"[Step 3/3] Classification in progress...")
    
    # Process the features for classification
    if verbose:
        print("Preparing features for classification...")
    
    # First ensure all columns are numeric
    numeric_df = results_df.select_dtypes(include=[np.number])
    
    if verbose:
        print(f"Using {numeric_df.shape[1]} numeric features for classification")
    
    # Average features across all structures to get a single feature vector
    X_test = np.mean(numeric_df, axis=0)
    X_test = pd.DataFrame(X_test).T
    
    # Some models might require specific feature sets, handle missing features
    feature_names = X_test.columns.tolist()
    if verbose:
        print(f"Feature vector size: {X_test.shape}")
    
    # Convert to numpy array for prediction
    X_test = np.array(X_test)
    
    # Get predictions and probabilities
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    
    class_names = ['Control', 'MCI', 'AD', 'VaD']
    label = class_names[int(preds[0])]
    
    # Create result dictionary
    results = {
        'prediction': label,
        'probabilities': {
            'Control': float(probs[0][0]),
            'MCI': float(probs[0][1]),
            'AD': float(probs[0][2]),
            'VaD': float(probs[0][3])
        }
    }
    
    # Save results to output directory
    results_file = os.path.join(output_dir, f"{filename_no_ext}_classification.txt")
    with open(results_file, 'w') as f:
        f.write(f"Classification Results for {filename_no_ext}:\n")
        f.write(f"Prediction: {label}\n\n")
        f.write("Probabilities:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}: {probs[0][i] * 100:.2f}%\n")
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    if verbose:
        print(f"Classification completed successfully in {int(time_taken)} seconds!")
        print(f"Prediction: {label}")
        print("Probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {probs[0][i] * 100:.2f}%")
        print(f"Results saved to {results_file}")
    
    # Clean up temporary files
    if clean:
        if verbose:
            print("Cleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to clean up temporary directory: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='MRI Brain Scan Classification Tool')
    parser.add_argument('--input', '-i', required=True, 
                        help='Path to input MRI file (.nii or .nii.gz)')
    parser.add_argument('--output', '-o', default='./output', 
                        help='Directory to store outputs')
    parser.add_argument('--model', '-m', default='./model_rf.pkl', 
                        help='Path to classification model (.pkl)')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Print detailed progress')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output results as JSON (even with verbose mode)')
    parser.add_argument('--keep-temp', '-k', action='store_true',
                        help='Keep temporary files (for debugging)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        return 1
    
    # Check if input file is a valid NIfTI file
    if not args.input.endswith(('.nii', '.nii.gz')):
        print(f"Error: Input file must be a NIfTI file (.nii or .nii.gz)", file=sys.stderr)
        return 1
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found", file=sys.stderr)
        return 1
    
    try:
        if args.verbose:
            print(f"Starting MRI classification pipeline...")
            print(f"Input file: {args.input}")
            print(f"Output directory: {args.output}")
            print(f"Model: {args.model}")
        
        # Process the MRI scan
        results = process_mri(args.input, args.output, args.model, args.verbose, not args.keep_temp)
        
        # Print results in JSON format if requested
        if args.json or not args.verbose:
            import json
            print(json.dumps(results, indent=2))
        
        if args.verbose:
            print(f"Classification completed successfully!")
            
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())