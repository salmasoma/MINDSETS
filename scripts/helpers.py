import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
# Structure Aggregator - First Compartment
###############################################################################
class StructureAggregator(nn.Module):
    """First compartment: Aggregates data across structures for each patient"""
    def __init__(self, feature_dim=107, num_structures=32, output_dim=107):
        super(StructureAggregator, self).__init__()
        self.feature_dim = feature_dim
        self.num_structures = num_structures
        self.output_dim = output_dim
        
        # Structure-aware transformation layer
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Attention mechanism for weighted structure aggregation
        self.attention = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, structure_indices=None):
        """
        Aggregate features across structures with attention
        
        Args:
            x: Tensor of shape [batch_size, num_structures, feature_dim]
            structure_indices: Optional tensor of shape [batch_size, num_structures]
                               with indices of each structure
        
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        batch_size, num_structures, feature_dim = x.shape
        
        # Apply transformation to each structure
        # Reshape to apply transformation to all structures at once
        x_flat = x.view(-1, feature_dim)  # [batch_size * num_structures, feature_dim]
        transformed = self.transform(x_flat)  # [batch_size * num_structures, output_dim]
        transformed = transformed.view(batch_size, num_structures, self.output_dim)  # [batch_size, num_structures, output_dim]
        
        # Calculate attention weights
        attn_scores = self.attention(transformed)  # [batch_size, num_structures, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch_size, num_structures, 1]
        
        # Apply attention weights
        weighted_features = transformed * attn_weights  # [batch_size, num_structures, output_dim]
        
        # Sum across structures
        aggregated = weighted_features.sum(dim=1)  # [batch_size, output_dim]
        
        return aggregated

###############################################################################
# Convolutional Feature Generator - Second Compartment
###############################################################################
class ConvFeatureGenerator(nn.Module):
    def __init__(self, num_fields, embedding_dim,
                 channels=[14, 16, 18, 20],
                 kernel_heights=[7, 7, 7, 7],
                 pooling_sizes=[2, 2, 2, 2],
                 recombined_channels=[2, 2, 2, 2],
                 activation="gelu",
                 batch_norm=True):
        super(ConvFeatureGenerator, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        conv_layers = []
        recombine_layers = []
        self.channels = [1] + channels  # input channel = 1
        input_height = num_fields
        
        # Store dimensions for debugging
        self.input_heights = [input_height]
        self.input_dims = []
        self.output_dims = []
        
        # Create convolutional layers for feature generation
        for i in range(1, len(self.channels)):
            in_channel = self.channels[i - 1]
            out_channel = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            pooling_size = pooling_sizes[i - 1]
            recombined_channel = recombined_channels[i - 1]
            
            # Convolutional block
            conv_block = [
                nn.Conv2d(in_channel, out_channel,
                         kernel_size=(kernel_height, 1),
                         padding=(int((kernel_height - 1) / 2), 0))
            ]
            if batch_norm:
                conv_block.append(nn.BatchNorm2d(out_channel))
            conv_block.extend([
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.MaxPool2d((pooling_size, 1), 
                           padding=(input_height % pooling_size, 0))
            ])
            conv_layers.append(nn.Sequential(*conv_block))
            
            # Calculate dimensions for recombination layer
            input_height = int(np.ceil(input_height / pooling_size))
            self.input_heights.append(input_height)
            
            # Calculate input and output dimensions for recombination
            input_dim = input_height * embedding_dim * out_channel
            output_dim = input_height * embedding_dim * recombined_channel
            
            self.input_dims.append(input_dim)
            self.output_dims.append(output_dim)
            
            # Recombination layer
            recombine_block = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU() if activation == "gelu" else nn.ReLU()
            )
            recombine_layers.append(recombine_block)
        
        self.conv_layers = nn.ModuleList(conv_layers)
        self.recombine_layers = nn.ModuleList(recombine_layers)
    
    def forward(self, x):
        # Input shape: [batch_size, 1, num_fields, embedding_dim]
        batch_size = x.size(0)
        conv_out = x
        new_features = []
        
        # Generate new features through conv and recombination
        for i in range(len(self.channels) - 1):
            # Apply convolution
            conv_out = self.conv_layers[i](conv_out)
            
            # Flatten for recombination
            # Reshape to preserve batch dimension
            flatten_out = conv_out.view(batch_size, -1)
            
            # Apply recombination
            recombine_out = self.recombine_layers[i](flatten_out)
            
            # Reshape to [batch_size, height, embedding_dim]
            height = self.input_heights[i + 1]
            recombined_channels = self.output_dims[i] // (height * self.embedding_dim)
            new_shape = (batch_size, height * recombined_channels, self.embedding_dim)
            reshaped = recombine_out.view(new_shape)
            
            new_features.append(reshaped)
        
        # Combine all new features
        new_feature_emb = torch.cat(new_features, dim=1)
        return new_feature_emb

###############################################################################
# FGCNN Model - Full Model with 3 Compartments
###############################################################################
class FGCNNModel(nn.Module):
    def __init__(self, input_dim=107, num_classes=4,
                 embedding_dim=128,
                 num_fields=32,  # 32 structures per patient
                 num_structures=32,  # Number of possible structure types
                 structure_embedding_dim=16,  # Embedding dimension for structures
                 channels=[32, 64, 128, 256],
                 kernel_heights=[5, 5, 5, 5],
                 pooling_sizes=[2, 2, 2, 2],
                 recombined_channels=[4, 4, 4, 4],
                 dnn_hidden_units=[2048, 1024, 512, 256],
                 dropout_rate=0.4):
        super(FGCNNModel, self).__init__()
        
        # Store dimensions
        self.input_dim = input_dim  # Should be exactly 107 radiomics features
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields  # Should be 32 structures per patient
        self.structure_embedding_dim = structure_embedding_dim
        
        # Structure embedding layer
        self.structure_embedding = nn.Embedding(
            num_embeddings=num_structures,
            embedding_dim=structure_embedding_dim
        )
        
        # First compartment: Structure Aggregator
        self.structure_aggregator = StructureAggregator(
            feature_dim=input_dim,
            num_structures=num_fields,
            output_dim=embedding_dim
        )
        
        # Feature generation component
        self.feature_generator = ConvFeatureGenerator(
            num_fields=1,  # Now working with aggregated data (1 structure per patient)
            embedding_dim=embedding_dim,
            channels=channels,
            kernel_heights=kernel_heights,
            pooling_sizes=pooling_sizes,
            recombined_channels=recombined_channels
        )
        
        # Calculate total features dimension
        self.total_features = 1  # Start with 1 (aggregated structure)
        input_height = 1
        for i in range(len(channels)):
            input_height = int(np.ceil(input_height / pooling_sizes[i]))
            self.total_features += input_height * recombined_channels[i]
        
        # Deep classifier with residual connections
        classifier_input_dim = self.total_features * embedding_dim
        
        # Initial dimension reduction
        self.dim_reduction = nn.Sequential(
            nn.Linear(classifier_input_dim, dnn_hidden_units[0]),
            nn.LayerNorm(dnn_hidden_units[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Main classifier layers
        self.classifier_layers = nn.ModuleList()
        current_dim = dnn_hidden_units[0]
        
        for next_dim in dnn_hidden_units[1:]:
            self.classifier_layers.append(
                nn.Sequential(
                    nn.Linear(current_dim, next_dim),
                    nn.LayerNorm(next_dim),
                    nn.GELU(),
                    nn.Dropout(dropout_rate)
                )
            )
            current_dim = next_dim
        
        # Final classification layer
        self.final_classifier = nn.Linear(dnn_hidden_units[-1], num_classes)
    
    def forward(self, features, structure_indices=None):
        """
        Forward pass with both radiomics features and structure indicators
        
        Args:
            features: Tensor of shape [batch_size, num_structures, input_dim]
                      Contains the radiomics features (should be exactly 107 features)
            structure_indices: Tensor of shape [batch_size, num_structures]
                               Contains the structure type indicators (optional)
                               
        Returns:
            Tensor of shape [batch_size, num_classes] with class logits
        """
        # Check input dimensions
        batch_size = features.size(0)
        num_structures = features.size(1)
        feature_dim = features.size(2)
        
        # Verify feature dimensions
        if feature_dim != self.input_dim:
            print(f"Warning: Feature dimension {feature_dim} doesn't match expected {self.input_dim}")
            # Fix feature dimension
            if feature_dim > self.input_dim:
                features = features[:, :, :self.input_dim]
            elif feature_dim < self.input_dim:
                padding = torch.zeros(batch_size, num_structures, self.input_dim - feature_dim, device=features.device)
                features = torch.cat([features, padding], dim=2)
        
        # Create structure indices if not provided
        if structure_indices is None:
            structure_indices = torch.zeros(batch_size, num_structures, dtype=torch.long, device=features.device)
        
        # First compartment: Aggregate structures
        aggregated = self.structure_aggregator(features, structure_indices)  # [batch_size, embedding_dim]
        
        # Reshape to [batch_size, 1, embedding_dim] for feature generator
        aggregated = aggregated.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Apply feature generator to aggregated representation
        # First reshape to [batch_size, 1, 1, embedding_dim] for ConvFeatureGenerator
        embedded = aggregated.unsqueeze(1)  # [batch_size, 1, 1, embedding_dim]
        new_features = self.feature_generator(embedded)  # [batch_size, num_new_fields, embedding_dim]
        
        # Combine features
        combined_features = torch.cat([aggregated, new_features], dim=1)  # [batch_size, 1 + num_new_fields, embedding_dim]
        flattened = combined_features.view(batch_size, -1)  # [batch_size, (1 + num_new_fields) * embedding_dim]
        
        # Initial dimension reduction
        x = self.dim_reduction(flattened)
        
        # Progressive classification with skip connections
        for layer in self.classifier_layers:
            identity = x
            x = layer(x)
            # Only add residual if dimensions match
            if x.size(-1) == identity.size(-1):
                x = x + identity
        
        # Final classification
        logits = self.final_classifier(x)
        
        return logits

###############################################################################
# Load trained model and necessary components
###############################################################################
def load_model(model_path, device=None):
    """
    Load a saved StructureAwareClassifier model
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model to (cpu/cuda)
        
    Returns:
        model: Loaded model
        label_encoder: Label encoder for class labels
        structure_encoder: Label encoder for structure types
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model components
    label_encoder = checkpoint.get('label_encoder')
    structure_encoder = checkpoint.get('structure_encoder')
    num_classes = checkpoint.get('num_classes', 4)  # Default to 4 if not found
    
    feature_dim = 107  # Default feature dimension
    
    # Determine feature dimension from model state dict
    for key, value in checkpoint['model_state_dict'].items():
        if 'structure_aggregator.transform.0.weight' in key:
            feature_dim = value.size(1)
            print(f"Detected feature dimension: {feature_dim}")
            break
    
    # Create model with correct architecture
    model = FGCNNModel(
        input_dim=feature_dim,
        num_classes=num_classes,
        embedding_dim=128,
        num_fields=32,  # 32 structures per patient
        num_structures=32,  # Number of possible structure types
        structure_embedding_dim=16,  # Embedding dimension for structures
        channels=[32, 64, 128, 256],
        kernel_heights=[5, 5, 5, 5],
        pooling_sizes=[2, 2, 2, 2],
        recombined_channels=[4, 4, 4, 4],
        dnn_hidden_units=[2048, 1024, 512, 256],
        dropout_rate=0.3
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Number of classes: {num_classes}")
    if label_encoder is not None:
        print(f"Class labels: {label_encoder.classes_}")
    
    return model, label_encoder, structure_encoder
