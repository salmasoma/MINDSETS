import argparse
import os
import time
import sys
from pathlib import Path
import warnings
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import joblib
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from helpers import load_model

try:
    from predict_synthseg import predict
except ImportError:
    print("Warning: Could not import SynthSeg predict function. Make sure it's available in your path.")
    print("Continuing with the assumption that segmentation files are provided.")

# Ignore all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="matplotlib")

###############################################################################
# MRI Processing and Feature Extraction
###############################################################################
def process_mri(input_file, output_dir, segmentation_model_path=None, verbose=False, clean=True):
    """
    Process an MRI scan through segmentation and feature extraction
    
    Args:
        input_file (str): Path to input MRI file (.nii or .nii.gz)
        output_dir (str): Directory to store outputs
        segmentation_model_path (str): Path to the segmentation model
        verbose (bool): Whether to print detailed progress
        clean (bool): Whether to clean temporary files after processing
    
    Returns:
        pd.DataFrame: Extracted radiomics features
    """
    start_time = time.time()
    
    # Define labels and structures - these should match what was used during training
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
    
    # Get filename info
    filename = os.path.basename(input_file)
    filename_no_ext = Path(filename).stem
    if filename_no_ext.endswith('.nii'):
        filename_no_ext = filename_no_ext[:-4]
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if segmentation model path variable exists in scope
    if 'predict' in globals():
        if verbose:
            print(f"[Step 1/2] Running Segmentation model...")
        
        # Run segmentation directly from source to output directory
        volumes_file = os.path.join(output_dir, f"{filename_no_ext}_volumes.csv")
        
        # Get base directory for model paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "..", "models")
        data_dir = os.path.join(base_dir, "..", "data", "labels_classes_priors")
        
        # Default paths for segmentation model and related files
        seg_model_path = os.path.join(model_dir, "synthseg_2.0.h5")
        seg_labels_path = os.path.join(data_dir, "synthseg_segmentation_labels_2.0.npy")
        seg_names_path = os.path.join(data_dir, "synthseg_segmentation_names_2.0.npy")
        denoiser_labels_path = os.path.join(data_dir, "synthseg_denoiser_labels_2.0.npy")
        parc_model_path = os.path.join(model_dir, "synthseg_parc_2.0.h5")
        parc_labels_path = os.path.join(data_dir, "synthseg_parcellation_labels.npy")
        parc_names_path = os.path.join(data_dir, "synthseg_parcellation_names.npy")
        qc_model_path = os.path.join(model_dir, "synthseg_qc_2.0.h5")
        qc_labels_path = os.path.join(data_dir, "synthseg_qc_labels_2.0.npy")
        qc_names_path = os.path.join(data_dir, "synthseg_qc_names_2.0.npy")
        topo_classes_path = os.path.join(data_dir, "synthseg_topological_classes_2.0.npy")
        
        # Override with provided segmentation model path if available
        if segmentation_model_path is not None:
            seg_model_path = segmentation_model_path
        
        # Run SynthSeg prediction
        predict(path_images=input_file,
                path_segmentations=output_dir,
                path_model_segmentation=seg_model_path,
                labels_segmentation=seg_labels_path,
                robust=False,
                fast=True,
                v1=False,
                do_parcellation=False,
                n_neutral_labels=19,
                names_segmentation=seg_names_path,
                labels_denoiser=denoiser_labels_path,
                path_posteriors=None,
                path_resampled=None,
                path_volumes=volumes_file,
                path_model_parcellation=parc_model_path,
                labels_parcellation=parc_labels_path,
                names_parcellation=parc_names_path,
                path_model_qc=qc_model_path,
                labels_qc=qc_labels_path,
                path_qc_scores=None,
                names_qc=qc_names_path,
                cropping=None,
                topology_classes=topo_classes_path,
                ct=False)
        
        # Find the segmentation file
        mask_file = os.path.join(output_dir, f"{filename_no_ext}_synthseg.nii.gz")
    else:
        # If predict function is not available, check if segmentation file exists
        mask_file = os.path.join(output_dir, f"{filename_no_ext}_synthseg.nii.gz")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Segmentation file not found: {mask_file} and SynthSeg is not available")
    
    if verbose:
        print(f"[Step 2/2] Extracting Radiomics...")
    
    # Extract radiomics features
    if verbose:
        print("Initializing radiomics feature extractor...")
    
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    
    # Load images
    mri_image = sitk.ReadImage(input_file)
    mask_image = sitk.ReadImage(mask_file)
    
    if verbose:
        print(f"MRI Image dimensions: {mri_image.GetSize()}")
        print(f"Mask Image dimensions: {mask_image.GetSize()}")
    
    # Resample mask to match the size of the MRI image if needed
    if mask_image.GetSize() != mri_image.GetSize():
        if verbose:
            print("Resampling mask to match MRI dimensions...")
        
        mask_image = sitk.Resample(mask_image, mri_image, sitk.Transform(), 
                                   sitk.sitkNearestNeighbor, 0.0, mask_image.GetPixelID())
        
        if verbose:
            print(f"Resampled mask dimensions: {mask_image.GetSize()}")
    
    # Extract features for each brain structure
    results_list = []
    if verbose:
        print(f"Extracting features for {len(labels_structures)} brain structures...")
    
    # Store structure labels
    structure_labels = []
    
    for idx, (label, structure) in enumerate(labels_structures.items()):
        if verbose:
            print(f"  Processing {structure} (label {label})...")
        try:
            features = extractor.execute(mri_image, mask_image, label)
            features['Label'] = label
            structure_labels.append(structure)
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
    
    # Rename 'Label' to 'Structure' to match expected format for classifier
    results_df = results_df.rename(columns={'Label': 'Structure'})
    
    # Reorder the structure column to the start
    results_df = results_df[["Structure"] + [col for col in results_df.columns if col != "Structure"]]
    
    # Save extracted features for reference
    features_file = os.path.join(output_dir, f"{filename_no_ext}_radiomics.csv")
    results_df.to_csv(features_file, index=False)
    
    if verbose:
        print(f"Extracted {len(results_df.columns)} features across {len(results_list)} structures")
        print(f"Features saved to {features_file}")
    
    # Clean up temporary files if requested
    if clean:
        if verbose:
            print("Cleaning up temporary files...")
        # Keep mask_file and features_file, but remove other intermediate files
        temp_files = [f for f in os.listdir(output_dir) if f.endswith('.nii.gz') and f != os.path.basename(mask_file)]
        for temp_file in temp_files:
            try:
                os.remove(os.path.join(output_dir, temp_file))
            except:
                pass
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    if verbose:
        print(f"MRI processing completed in {int(time_taken)} seconds!")
    
    # Add PatientID column for compatibility with classifier
    results_df['PatientID'] = filename_no_ext
    
    # Ensure exactly 32 structures (as expected by the model)
    if len(results_df) > 32:
        if verbose:
            print(f"More than 32 structures found ({len(results_df)}). Keeping only first 32.")
        results_df = results_df.iloc[:32]
    elif len(results_df) < 32:
        if verbose:
            print(f"Fewer than 32 structures found ({len(results_df)}). Adding dummy structures.")
        # Add dummy structures by duplicating existing ones
        missing = 32 - len(results_df)
        dummy_df = results_df.iloc[:missing].copy()
        # Adjust structure names to indicate they are duplicates
        for i, idx in enumerate(dummy_df.index):
            dummy_df.loc[idx, 'Structure'] = f"dummy_{i+1}_{dummy_df.loc[idx, 'Structure']}"
        
        results_df = pd.concat([results_df, dummy_df], ignore_index=True)
    
    return results_df

###############################################################################
# Feature Preprocessing for Structure-Aware Classifier
###############################################################################
def preprocess_radiomics_features(df, feature_cols=None):
    """
    Preprocess radiomics features for input to the structure-aware classifier
    
    Args:
        df: DataFrame with radiomics features
        feature_cols: List of feature columns to use (if None, will be auto-detected)
        
    Returns:
        df_processed: Preprocessed DataFrame ready for the classifier
    """
    df_processed = df.copy()
    
    # Identify feature columns if not provided
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in ['PatientID', 'Structure']]
    
    # Ensure all feature columns are numeric
    for col in feature_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Handle missing values
    for col in feature_cols:
        missing_count = df_processed[col].isnull().sum()
        if missing_count > 0:
            # Fill missing values with column means
            df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
    
    # Replace inf values
    for col in feature_cols:
        if (df_processed[col].abs() == np.inf).any():
            # Replace inf with large values
            max_val = df_processed[col][np.isfinite(df_processed[col])].max()
            min_val = df_processed[col][np.isfinite(df_processed[col])].min()
            df_processed.loc[df_processed[col] == np.inf, col] = max_val * 10
            df_processed.loc[df_processed[col] == -np.inf, col] = min_val * 10
    
    # Standardize features (important for model performance)
    scaler = StandardScaler()
    # Standardize each feature column
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
    
    # Final check for any non-finite values
    for col in feature_cols:
        non_finite = (~np.isfinite(df_processed[col])).sum()
        if non_finite > 0:
            df_processed[col] = np.nan_to_num(df_processed[col], nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Verify we have exactly 107 feature columns (as expected by the model)
    if len(feature_cols) > 107:
        print(f"Warning: Found {len(feature_cols)} features, but model expects 107. Trimming to first 107.")
        feature_cols = feature_cols[:107]
        df_processed = df_processed[['PatientID', 'Structure'] + feature_cols]
    elif len(feature_cols) < 107:
        print(f"Warning: Found only {len(feature_cols)} features, but model expects 107. Adding dummy features.")
        # Add dummy features
        for i in range(len(feature_cols), 107):
            col_name = f"dummy_feature_{i}"
            df_processed[col_name] = 0.0
    
    return df_processed

###############################################################################
# Structure-Aware Model Inference
###############################################################################
def run_inference(model, df, label_encoder, structure_encoder, device):
    """
    Run inference using the structure-aware classifier
    
    Args:
        model: Loaded StructureAwareClassifier model
        df: Preprocessed DataFrame with radiomics features
        label_encoder: Label encoder for class labels
        structure_encoder: Label encoder for structure types
        device: Device to run inference on
        
    Returns:
        dict: Classification results with class probabilities
    """
    # Set model to evaluation mode
    model.eval()
    
    # Prepare features and structure indices
    feature_cols = [col for col in df.columns if col not in ['PatientID', 'Structure']]
    features = df[feature_cols].values.astype(np.float32)
    
    # Get structure indices - map structure names to indices using structure encoder
    # If structure not in encoder, assign a default index (0)
    structure_indices = []
    for structure in df['Structure']:
        try:
            if structure in structure_encoder.classes_:
                idx = structure_encoder.transform([structure])[0]
            else:
                # If structure not found, use first structure type as default
                idx = 0
            structure_indices.append(idx)
        except:
            # Fallback for any issues with structure encoder
            structure_indices.append(0)
    
    # Convert to tensors
    features = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension [1, num_structures, num_features]
    structure_indices = torch.LongTensor(structure_indices).unsqueeze(0)  # Add batch dimension [1, num_structures]
    
    # Move to device
    features = features.to(device)
    structure_indices = structure_indices.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(features, structure_indices)
        
        # Get probabilities
        probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
        
        # Get predicted class
        pred_class_idx = torch.argmax(outputs, dim=1).item()
        pred_class = label_encoder.inverse_transform([pred_class_idx])[0]
        
        # Calculate confidence
        confidence = probabilities[pred_class_idx]
    
    # Create result dictionary
    results = {
        'prediction': pred_class,
        'confidence': float(confidence),
        'probabilities': {}
    }
    
    # Add class probabilities
    for i, class_name in enumerate(label_encoder.classes_):
        results['probabilities'][class_name] = float(probabilities[i])
    
    return results

###############################################################################
# Main Function
###############################################################################
def main():
    parser = argparse.ArgumentParser(description='MRI Brain Scan Classification Tool')
    parser.add_argument('--input', '-i', required=True, 
                        help='Path to input MRI file (.nii or .nii.gz)')
    parser.add_argument('--output', '-o', default='./output', 
                        help='Directory to store outputs')
    parser.add_argument('--model', '-m', required=True, 
                        help='Path to structure-aware classification model (.pt)')
    parser.add_argument('--seg-model', '-s', default=None,
                        help='Path to segmentation model (optional)')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU for inference if available')
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
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    try:
        if args.verbose:
            print(f"Starting MRI classification pipeline...")
            print(f"Input file: {args.input}")
            print(f"Output directory: {args.output}")
            print(f"Model: {args.model}")
            print(f"Device: {device}")
        
        # Step 1: Load the model
        if args.verbose:
            print("\n=== Step 1: Loading Model ===")
        
        model, label_encoder, structure_encoder = load_model(args.model, device)
        
        # Step 2: Process MRI and extract radiomics features
        if args.verbose:
            print("\n=== Step 2: Processing MRI and Extracting Features ===")
        
        radiomics_df = process_mri(
            input_file=args.input,
            output_dir=args.output,
            segmentation_model_path=args.seg_model,
            verbose=args.verbose,
            clean=not args.keep_temp
        )
        
        # Step 3: Preprocess features for the structure-aware classifier
        if args.verbose:
            print("\n=== Step 3: Preprocessing Features ===")
        
        preprocessed_df = preprocess_radiomics_features(radiomics_df)
        
        # Save preprocessed features for reference
        filename = os.path.basename(args.input)
        filename_no_ext = Path(filename).stem
        if filename_no_ext.endswith('.nii'):
            filename_no_ext = filename_no_ext[:-4]
        
        preprocessed_file = os.path.join(args.output, f"{filename_no_ext}_preprocessed.csv")
        preprocessed_df.to_csv(preprocessed_file, index=False)
        
        if args.verbose:
            print(f"Preprocessed features saved to {preprocessed_file}")
        
        # Step 4: Run inference with the structure-aware model
        if args.verbose:
            print("\n=== Step 4: Running Structure-Aware Classification ===")
        
        results = run_inference(
            model=model,
            df=preprocessed_df,
            label_encoder=label_encoder,
            structure_encoder=structure_encoder,
            device=device
        )
        
        # Step 5: Save results
        if args.verbose:
            print("\n=== Step 5: Saving Results ===")
        
        results_file = os.path.join(args.output, f"{filename_no_ext}_classification.txt")
        with open(results_file, 'w') as f:
            f.write(f"Classification Results for {filename_no_ext}:\n")
            f.write(f"Prediction: {results['prediction']}\n")
            f.write(f"Confidence: {results['confidence'] * 100:.2f}%\n\n")
            f.write("Probabilities:\n")
            for class_name, prob in results['probabilities'].items():
                f.write(f"{class_name}: {prob * 100:.2f}%\n")
        
        json_file = os.path.join(args.output, f"{filename_no_ext}_classification.json")
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if args.verbose:
            print(f"Results saved to {results_file} and {json_file}")
        
        # Print results
        if args.json or not args.verbose:
            print(json.dumps(results, indent=2))
        else:
            print("\n=== Classification Results ===")
            print(f"Prediction: {results['prediction']}")
            print(f"Confidence: {results['confidence'] * 100:.2f}%")
            print("\nProbabilities:")
            for class_name, prob in results['probabilities'].items():
                print(f"{class_name}: {prob * 100:.2f}%")
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
