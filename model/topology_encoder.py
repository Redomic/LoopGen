"""
Topology encoder for binding site characterization using persistent homology.

This module uses topological data analysis (TDA) to extract geometric features
from protein binding pockets, capturing cavity structure, loops, and voids
without requiring full 3D atomic coordinates.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Import TDA libraries
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceImage, PersistenceLandscape, Amplitude
    GIOTTO_AVAILABLE = True
except ImportError:
    logger.warning("giotto-tda not available - install with: pip install giotto-tda")
    GIOTTO_AVAILABLE = False


class TopologyFeatureExtractor:
    """
    Extract topological features from distance matrices using persistent homology.
    
    This class computes persistence diagrams and converts them to fixed-size
    feature vectors using persistence images or landscapes.
    """
    
    def __init__(
        self,
        homology_dimensions: List[int] = [0, 1, 2],
        n_bins: int = 50,
        pixel_size: float = 0.1,
        representation: str = 'image'
    ):
        """
        Args:
            homology_dimensions: Which homology dimensions to compute (0D, 1D, 2D)
            n_bins: Number of bins for persistence image (n_bins x n_bins grid)
            pixel_size: Pixel size for persistence image
            representation: 'image' or 'landscape' for vectorization method
        """
        if not GIOTTO_AVAILABLE:
            raise ImportError("giotto-tda is required. Install with: pip install giotto-tda")
        
        self.homology_dimensions = homology_dimensions
        self.n_bins = n_bins
        self.pixel_size = pixel_size
        self.representation = representation
        
        # Initialize Vietoris-Rips persistence computer
        self.persistence = VietorisRipsPersistence(
            metric='precomputed',
            homology_dimensions=homology_dimensions,
            n_jobs=1  # Single job for thread safety in DataLoader
        )
        
        # Initialize vectorization method
        if representation == 'image':
            self.vectorizer = PersistenceImage(
                n_bins=n_bins,
                sigma=pixel_size
            )
        elif representation == 'landscape':
            self.vectorizer = PersistenceLandscape(
                n_layers=5,
                n_bins=n_bins
            )
        else:
            raise ValueError(f"Unknown representation: {representation}")
        
        logger.info(f"Initialized TopologyFeatureExtractor with {homology_dimensions}D homology")
    
    def compute_persistence_diagrams(
        self, 
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute persistence diagrams from distance matrix.
        
        Args:
            distance_matrix: Pairwise distance matrix [n, n]
            
        Returns:
            Persistence diagrams array [1, n_points, 3]
            Format: (birth, death, homology_dimension) for each point
        """
        # Ensure distance matrix is 2D
        if len(distance_matrix.shape) != 2:
            raise ValueError(f"Expected 2D distance matrix, got shape {distance_matrix.shape}")
        
        # Add batch dimension for giotto-tda: [1, n, n]
        distance_matrix_batch = distance_matrix[np.newaxis, :, :]
        
        # Compute persistence diagrams
        try:
            diagrams = self.persistence.fit_transform(distance_matrix_batch)
            return diagrams
        except Exception as e:
            logger.warning(f"Failed to compute persistence diagrams: {e}")
            # Return empty diagram
            return np.zeros((1, 0, 3), dtype=np.float32)
    
    def vectorize_persistence_diagrams(
        self, 
        diagrams: np.ndarray
    ) -> np.ndarray:
        """
        Convert persistence diagrams to fixed-size feature vectors.
        
        Args:
            diagrams: Persistence diagrams [1, n_points, 3]
            
        Returns:
            Feature vector [feature_dim]
        """
        try:
            features = self.vectorizer.fit_transform(diagrams)
            # Shape: [1, n_bins, n_bins, n_homology_dims] for images
            # Flatten to 1D vector
            features_flat = features.reshape(features.shape[0], -1)
            return features_flat[0]  # Remove batch dimension
        except Exception as e:
            logger.warning(f"Failed to vectorize persistence diagrams: {e}")
            # Return zero vector
            feature_dim = (n_bins ** 2) * len(self.homology_dimensions)
            return np.zeros(feature_dim, dtype=np.float32)
    
    def extract_features(
        self, 
        distance_matrix: np.ndarray
    ) -> np.ndarray:
        """
        End-to-end feature extraction from distance matrix.
        
        Args:
            distance_matrix: Pairwise distance matrix [n, n]
            
        Returns:
            Topology feature vector [feature_dim]
        """
        # Compute persistence diagrams
        diagrams = self.compute_persistence_diagrams(distance_matrix)
        
        # Vectorize to fixed-size features
        features = self.vectorize_persistence_diagrams(diagrams)
        
        return features


class TopologyEncoder(nn.Module):
    """
    Neural network module for encoding binding site topology.
    
    Processes distance matrices through persistent homology and projects
    to model dimensionality for fusion with sequence embeddings.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        homology_dimensions: List[int] = [0, 1, 2],
        n_bins: int = 50,
        dropout: float = 0.1,
        representation: str = 'image'
    ):
        """
        Args:
            d_model: Model dimension for projection
            homology_dimensions: Which homology dimensions to compute
            n_bins: Number of bins for persistence images
            dropout: Dropout rate
            representation: 'image' or 'landscape'
        """
        super().__init__()
        
        self.d_model = d_model
        self.homology_dimensions = homology_dimensions
        self.n_bins = n_bins
        self.representation = representation
        
        # Initialize topology feature extractor
        if GIOTTO_AVAILABLE:
            self.feature_extractor = TopologyFeatureExtractor(
                homology_dimensions=homology_dimensions,
                n_bins=n_bins,
                representation=representation
            )
            
            # Calculate feature dimension
            if representation == 'image':
                self.feature_dim = (n_bins ** 2) * len(homology_dimensions)
            elif representation == 'landscape':
                self.feature_dim = 5 * n_bins * len(homology_dimensions)
            
            logger.info(f"Topology feature dimension: {self.feature_dim}")
        else:
            logger.warning("giotto-tda not available - TopologyEncoder will not function")
            self.feature_extractor = None
            self.feature_dim = 128  # Placeholder
        
        # MLP projection to model dimension
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection layer weights."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def extract_topology_features_batch(
        self, 
        distance_matrices: List[np.ndarray]
    ) -> torch.Tensor:
        """
        Extract topology features from batch of distance matrices.
        
        Args:
            distance_matrices: List of distance matrices, each [n_i, n_i]
            
        Returns:
            Batch of topology features [batch_size, feature_dim]
        """
        if not GIOTTO_AVAILABLE or self.feature_extractor is None:
            # Return zero features if TDA not available
            batch_size = len(distance_matrices)
            return torch.zeros(batch_size, self.feature_dim)
        
        batch_features = []
        
        for dist_matrix in distance_matrices:
            try:
                features = self.feature_extractor.extract_features(dist_matrix)
                batch_features.append(features)
            except Exception as e:
                logger.debug(f"Error extracting topology features: {e}")
                # Use zero features for failed extractions
                batch_features.append(np.zeros(self.feature_dim, dtype=np.float32))
        
        # Convert to tensor
        features_tensor = torch.from_numpy(np.stack(batch_features)).float()
        
        return features_tensor
    
    def forward(
        self, 
        topology_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Project topology features to model dimension.
        
        Args:
            topology_features: Pre-extracted topology features [batch, feature_dim]
            
        Returns:
            Projected topology embeddings [batch, d_model]
        """
        # Project to model dimension
        topology_embeddings = self.projection(topology_features)
        
        return topology_embeddings


class TopologyFusionLayer(nn.Module):
    """
    Fusion layer to combine sequence embeddings with topology features.
    
    Supports multiple fusion strategies: concatenation, addition, or gating.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        fusion_method: str = 'concat',
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            fusion_method: 'concat', 'add', or 'gate'
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            # Concatenate and project back to d_model
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            )
        elif fusion_method == 'add':
            # Simple addition with layer norm
            self.fusion = nn.LayerNorm(d_model)
        elif fusion_method == 'gate':
            # Gating mechanism
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.fusion = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize fusion layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        sequence_embeddings: torch.Tensor,
        topology_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse sequence and topology embeddings.
        
        Args:
            sequence_embeddings: [batch, seq_len, d_model]
            topology_embeddings: [batch, d_model]
            
        Returns:
            Fused embeddings [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = sequence_embeddings.size()
        
        # Broadcast topology to each position
        topology_broadcast = topology_embeddings.unsqueeze(1).expand(
            batch_size, seq_len, self.d_model
        )
        
        if self.fusion_method == 'concat':
            # Concatenate and project
            combined = torch.cat([sequence_embeddings, topology_broadcast], dim=-1)
            fused = self.fusion(combined)
        
        elif self.fusion_method == 'add':
            # Simple addition
            fused = self.fusion(sequence_embeddings + topology_broadcast)
        
        elif self.fusion_method == 'gate':
            # Gating mechanism
            combined = torch.cat([sequence_embeddings, topology_broadcast], dim=-1)
            gate = self.gate(combined)
            fused = self.fusion(
                gate * sequence_embeddings + (1 - gate) * topology_broadcast
            )
        
        return fused


if __name__ == "__main__":
    # Test topology encoder
    print("Testing TopologyEncoder...")
    
    if GIOTTO_AVAILABLE:
        # Create dummy distance matrix
        n_residues = 50
        coords = np.random.randn(n_residues, 3).astype(np.float32)
        dist_matrix = np.linalg.norm(
            coords[:, np.newaxis, :] - coords[np.newaxis, :, :], 
            axis=2
        )
        
        # Test feature extraction
        extractor = TopologyFeatureExtractor(
            homology_dimensions=[0, 1],
            n_bins=20
        )
        features = extractor.extract_features(dist_matrix)
        print(f"Extracted topology features shape: {features.shape}")
        
        # Test encoder
        encoder = TopologyEncoder(d_model=256, n_bins=20)
        features_tensor = torch.from_numpy(features).unsqueeze(0)
        embeddings = encoder(features_tensor)
        print(f"Topology embeddings shape: {embeddings.shape}")
        
        # Test fusion
        fusion = TopologyFusionLayer(d_model=256, fusion_method='concat')
        seq_emb = torch.randn(2, 100, 256)
        topo_emb = torch.randn(2, 256)
        fused = fusion(seq_emb, topo_emb)
        print(f"Fused embeddings shape: {fused.shape}")
        
        print("All tests passed!")
    else:
        print("giotto-tda not available - skipping tests")


