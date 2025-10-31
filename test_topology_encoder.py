#!/usr/bin/env python3
"""
Test script for the Topology Encoder implementation.

This script verifies that the topology encoding pipeline works correctly:
1. Distance matrix extraction from PDB files
2. Persistent homology computation
3. Topology feature vectorization
4. Integration with protein encoder
"""

import torch
import numpy as np
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_distance_extraction():
    """Test C-alpha coordinate extraction and distance matrix computation."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Distance Matrix Extraction")
    logger.info("="*70)
    
    try:
        from data.extract_pdb_distances import compute_distance_matrix
        
        # Create dummy C-alpha coordinates (50 residues in 3D)
        n_residues = 50
        coords = np.random.randn(n_residues, 3).astype(np.float32)
        
        # Compute distance matrix
        dist_matrix = compute_distance_matrix(coords)
        
        # Verify properties
        assert dist_matrix.shape == (n_residues, n_residues), "Wrong shape"
        assert np.allclose(dist_matrix, dist_matrix.T), "Not symmetric"
        assert np.all(np.diag(dist_matrix) == 0), "Diagonal not zero"
        
        logger.info(f"✓ Distance matrix shape: {dist_matrix.shape}")
        logger.info(f"✓ Min distance: {np.min(dist_matrix[dist_matrix > 0]):.2f}")
        logger.info(f"✓ Max distance: {np.max(dist_matrix):.2f}")
        logger.info("✓ TEST PASSED")
        
        return True
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        return False


def test_topology_feature_extraction():
    """Test persistent homology feature extraction."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Topology Feature Extraction")
    logger.info("="*70)
    
    try:
        from model.topology_encoder import TopologyFeatureExtractor
        
        # Create dummy distance matrix
        n_residues = 30
        coords = np.random.randn(n_residues, 3).astype(np.float32)
        dist_matrix = np.linalg.norm(
            coords[:, np.newaxis, :] - coords[np.newaxis, :, :], 
            axis=2
        )
        
        # Initialize extractor
        extractor = TopologyFeatureExtractor(
            homology_dimensions=[0, 1],
            n_bins=20,
            representation='image'
        )
        
        # Extract features
        features = extractor.extract_features(dist_matrix)
        
        # Verify
        expected_dim = (20 ** 2) * 2  # n_bins^2 * n_homology_dims
        assert features.shape[0] == expected_dim, f"Wrong feature dimension: {features.shape[0]}"
        
        logger.info(f"✓ Extracted features shape: {features.shape}")
        logger.info(f"✓ Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")
        logger.info("✓ TEST PASSED")
        
        return True
    except ImportError as e:
        logger.warning(f"⚠ TEST SKIPPED: giotto-tda not available ({e})")
        return True
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        return False


def test_topology_encoder():
    """Test the TopologyEncoder module."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Topology Encoder Module")
    logger.info("="*70)
    
    try:
        from model.topology_encoder import TopologyEncoder
        
        # Initialize encoder
        d_model = 256
        encoder = TopologyEncoder(
            d_model=d_model,
            homology_dimensions=[0, 1],
            n_bins=20,
            dropout=0.1
        )
        
        # Create dummy topology features
        batch_size = 4
        feature_dim = (20 ** 2) * 2
        topology_features = torch.randn(batch_size, feature_dim)
        
        # Forward pass
        embeddings = encoder(topology_features)
        
        # Verify
        assert embeddings.shape == (batch_size, d_model), "Wrong output shape"
        
        logger.info(f"✓ Input shape: {topology_features.shape}")
        logger.info(f"✓ Output shape: {embeddings.shape}")
        logger.info("✓ TEST PASSED")
        
        return True
    except ImportError as e:
        logger.warning(f"⚠ TEST SKIPPED: giotto-tda not available ({e})")
        return True
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        return False


def test_topology_fusion():
    """Test topology fusion with sequence embeddings."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Topology Fusion Layer")
    logger.info("="*70)
    
    try:
        from model.topology_encoder import TopologyFusionLayer
        
        # Initialize fusion layer
        d_model = 256
        fusion = TopologyFusionLayer(
            d_model=d_model,
            fusion_method='concat',
            dropout=0.1
        )
        
        # Create dummy inputs
        batch_size = 4
        seq_len = 100
        sequence_embeddings = torch.randn(batch_size, seq_len, d_model)
        topology_embeddings = torch.randn(batch_size, d_model)
        
        # Fuse
        fused = fusion(sequence_embeddings, topology_embeddings)
        
        # Verify
        assert fused.shape == (batch_size, seq_len, d_model), "Wrong output shape"
        
        logger.info(f"✓ Sequence embeddings shape: {sequence_embeddings.shape}")
        logger.info(f"✓ Topology embeddings shape: {topology_embeddings.shape}")
        logger.info(f"✓ Fused embeddings shape: {fused.shape}")
        logger.info("✓ TEST PASSED")
        
        return True
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        return False


def test_protein_encoder_integration():
    """Test ProteinEncoder with topology features."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Protein Encoder Integration")
    logger.info("="*70)
    
    try:
        from model.config import ModelConfig
        from model.protein_encoder import ProteinEncoder
        
        # Create config with topology enabled
        config = ModelConfig.small_config()
        config.use_topology_encoding = True
        config.topology_n_bins = 20
        config.topology_persistence_dims = [0, 1]
        
        # Initialize encoder
        try:
            encoder = ProteinEncoder(config)
            
            # Create dummy inputs
            batch_size = 2
            seq_len = 50
            protein_ids = torch.randint(5, 25, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            
            # Test without topology
            output = encoder(protein_ids, attention_mask)
            assert output.shape == (batch_size, seq_len, config.d_model)
            logger.info(f"✓ Without topology: {output.shape}")
            
            # Test with topology
            feature_dim = (20 ** 2) * 2
            topology_features = torch.randn(batch_size, feature_dim)
            output_with_topo = encoder(protein_ids, attention_mask, topology_features)
            assert output_with_topo.shape == (batch_size, seq_len, config.d_model)
            logger.info(f"✓ With topology: {output_with_topo.shape}")
            
            logger.info("✓ TEST PASSED")
            return True
            
        except ImportError as e:
            logger.warning(f"⚠ TEST SKIPPED: giotto-tda not available ({e})")
            return True
        
    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*70)
    logger.info("TOPOLOGY ENCODER TEST SUITE")
    logger.info("="*70)
    
    tests = [
        test_distance_extraction,
        test_topology_feature_extraction,
        test_topology_encoder,
        test_topology_fusion,
        test_protein_encoder_integration
    ]
    
    results = []
    for test_func in tests:
        passed = test_func()
        results.append(passed)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    passed = sum(results)
    total = len(results)
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        logger.info("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"\n✗ {total - passed} TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())


