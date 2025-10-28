#!/bin/bash

# =============================================================================
# Protein Conditioning Quick Start Script
# =============================================================================
# This script provides example commands for the complete protein conditioning
# workflow, from data preparation to model training and evaluation.

set -e  # Exit on error

echo "========================================="
echo "Protein Conditioning Quick Start"
echo "========================================="
echo

# Activate conda environment
echo "Step 1: Activating LoopGen environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate LoopGen
echo "✓ Environment activated"
echo

# Step 2: Check if CrossDock data is available
echo "Step 2: Checking for CrossDock data..."
if [ ! -f "data/output/positive_pairs.csv" ]; then
    echo "⚠ CrossDock data not found. Please run data/download_crossdock.sh first."
    echo "Example:"
    echo "  bash data/download_crossdock.sh --download"
    exit 1
fi
echo "✓ CrossDock data found"
echo

# Step 3: Extract pocket sequences
echo "Step 3: Extracting protein pocket sequences..."
if [ ! -f "data/output/protein_ligand_training.csv" ]; then
    echo "Running pocket extraction (this may take a while)..."
    python data/extract_pockets.py \
        --input data/output/positive_pairs.csv \
        --output data/output/protein_ligand_training.csv \
        --cutoff 10.0 \
        --min-length 10 \
        --max-length 500 \
        --max-pairs 100000
    echo "✓ Pocket extraction complete"
else
    echo "✓ Pocket sequences already extracted"
fi
echo

# Step 4: Check available data
echo "Step 4: Checking extracted data..."
if [ -f "data/output/protein_ligand_training.csv" ]; then
    NUM_PAIRS=$(wc -l < data/output/protein_ligand_training.csv)
    NUM_PAIRS=$((NUM_PAIRS - 1))  # Subtract header
    echo "✓ Found $NUM_PAIRS protein-ligand pairs"
    
    # Show sample
    echo "Sample data:"
    head -n 3 data/output/protein_ligand_training.csv
else
    echo "⚠ No training data found!"
    exit 1
fi
echo

# Step 5: Training options
echo "Step 5: Training Options"
echo "----------------------------------------"
echo

echo "Option A: Train with protein conditioning (RECOMMENDED)"
echo "This trains a model that generates molecules for specific proteins:"
echo
cat << 'EOF'
python train.py \
  --use_protein_conditioning \
  --protein_ligand_data_path data/output/protein_ligand_training.csv \
  --output_dir checkpoints \
  --model_size standard \
  --batch_size 128 \
  --num_epochs 80 \
  --learning_rate 3e-4 \
  --protein_max_seq_len 512 \
  --protein_encoder_layers 6 \
  --protein_encoder_heads 8 \
  --cross_attention_freq 1 \
  --use_amp \
  --num_workers 8 \
  --generate_interval 5
EOF
echo
echo "----------------------------------------"
echo

echo "Option B: Quick test with small model and subset (FAST)"
echo "For testing the implementation quickly:"
echo
cat << 'EOF'
python train.py \
  --use_protein_conditioning \
  --protein_ligand_data_path data/output/protein_ligand_training.csv \
  --output_dir checkpoints \
  --model_size small \
  --batch_size 64 \
  --num_epochs 10 \
  --dataset_subset_size 10000 \
  --val_set_size 1000 \
  --use_amp \
  --num_workers 4 \
  --generate_interval 2
EOF
echo
echo "----------------------------------------"
echo

# Step 6: Evaluation
echo "Step 6: After Training - Evaluation"
echo "----------------------------------------"
echo "To evaluate the trained model:"
echo
cat << 'EOF'
python evaluate_protein_conditioned.py \
  --checkpoint checkpoints/run_YYYYMMDD_HHMMSS/best_model.pt \
  --vocab checkpoints/vocab.json \
  --test_data data/output/protein_ligand_training.csv \
  --model_size standard \
  --num_proteins 20 \
  --samples_per_protein 20 \
  --output evaluation_results.json
EOF
echo
echo "========================================="
echo

# Interactive prompt
echo "Would you like to start training now? (y/n)"
read -r REPLY
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting training with small model for quick test..."
    echo "You can modify the command above for production training."
    echo
    
    python train.py \
        --use_protein_conditioning \
        --protein_ligand_data_path data/output/protein_ligand_training.csv \
        --output_dir checkpoints \
        --model_size small \
        --batch_size 64 \
        --num_epochs 10 \
        --dataset_subset_size 10000 \
        --val_set_size 1000 \
        --use_amp \
        --num_workers 4 \
        --generate_interval 2
    
    echo
    echo "✓ Training complete!"
    echo "Check checkpoints/ directory for model files"
else
    echo "Skipping training. Copy one of the commands above to train manually."
fi

echo
echo "========================================="
echo "Quick Start Complete!"
echo "========================================="
echo
echo "Next steps:"
echo "1. Train model with protein conditioning (see options above)"
echo "2. Monitor training in checkpoints/run_*/training.log"
echo "3. Evaluate model with evaluate_protein_conditioned.py"
echo "4. Generate molecules for specific proteins (see PROTEIN_CONDITIONING_GUIDE.md)"
echo
echo "For detailed documentation, see:"
echo "  - PROTEIN_CONDITIONING_GUIDE.md (complete usage guide)"
echo "  - protein-conditioning-implementation.plan.md (implementation plan)"
echo



