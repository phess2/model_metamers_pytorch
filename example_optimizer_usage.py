#!/usr/bin/env python3
"""
Example script demonstrating how to use different optimizers for metamer generation.

This script shows how to run metamer generation with either Adam or Muon optimizers.
"""

import subprocess
import sys

def run_metamer_with_optimizer(optimizer="adam", sound_index=0):
    """
    Run metamer generation with the specified optimizer.
    
    Args:
        optimizer (str): Either "adam" or "muon"
        sound_index (int): Index of the sound to process
    """
    if optimizer not in ["adam", "muon"]:
        raise ValueError("Optimizer must be either 'adam' or 'muon'")
    
    print(f"Running metamer generation with {optimizer.upper()} optimizer...")
    
    # Command to run the metamer generation script
    cmd = [
        "python", 
        "model_analysis_folders/visual_networks/resnet50/make_metamers_imagenet_16_category_val_400_only_save_metamer_layers.py",
        str(sound_index),
        "-I", "100",  # Reduced iterations for testing
        "-N", "2",    # Reduced repetitions for testing
        "-O", optimizer
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully completed with {optimizer} optimizer!")
        print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running with {optimizer} optimizer:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Main function to demonstrate both optimizers."""
    print("=== Metamer Generation Optimizer Comparison ===\n")
    
    # Test with Adam optimizer
    print("1. Testing Adam optimizer...")
    adam_success = run_metamer_with_optimizer("adam", sound_index=0)
    
    print("\n" + "="*50 + "\n")
    
    # Test with Muon optimizer
    print("2. Testing Muon optimizer...")
    muon_success = run_metamer_with_optimizer("muon", sound_index=0)
    
    print("\n" + "="*50 + "\n")
    
    # Summary
    print("Summary:")
    print(f"Adam optimizer: {'✓ Success' if adam_success else '✗ Failed'}")
    print(f"Muon optimizer: {'✓ Success' if muon_success else '✗ Failed'}")
    
    if muon_success:
        print("\nNote: If Muon optimizer is not installed, the system will automatically")
        print("fall back to Adam optimizer with a warning message.")

if __name__ == "__main__":
    main() 