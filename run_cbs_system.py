"""
Master Execution Script for CBS System
Interactive menu to coordinate all system components
"""

import os
import sys
import torch
from pathlib import Path
import config
from cbs_model_enhanced import create_model
from waymo_tfexample_loader import get_data_loaders
from primitive_detector_v2 import PrimitiveDetector, test_primitive_detection
from train_enhanced import train
from visualize_multi_agent import TrajectoryVisualizer, generate_video_from_data
from demo_real_waymo import InteractiveDemo, run_demo
import evaluate


def print_menu():
    """Print main menu"""
    print("\n" + "="*60)
    print("Compositional Behavior Synthesis (CBS) System")
    print("="*60)
    print("\nMain Menu:")
    print("  1. Test Primitive Detection")
    print("  2. Test Model Architecture")
    print("  3. Train CBS Model")
    print("  4. Generate Prediction Videos")
    print("  5. Run Complete Pipeline")
    print("  6. Quick Demo")
    print("  7. Evaluate Model")
    print("  8. Exit")
    print("="*60)


def test_primitive_detection_menu():
    """Test primitive detection system"""
    print("\n" + "-"*60)
    print("Testing Primitive Detection System")
    print("-"*60)
    
    try:
        test_primitive_detection()
        print("\n✓ Primitive detection test completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during primitive detection test: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")


def test_model_architecture():
    """Test CBS model architecture"""
    print("\n" + "-"*60)
    print("Testing CBS Model Architecture")
    print("-"*60)
    
    try:
        device = config.get_device()
        print(f"Device: {device}")
        
        model = create_model(device)
        print(f"\nModel created successfully!")
        print(f"Total parameters: {model.count_parameters():,}")
        print(f"Target: ~8,500,000")
        
        # Test forward pass
        batch_size = 2
        history = torch.randn(
            batch_size,
            config.MODEL_CONFIG["history_length"],
            config.MODEL_CONFIG["input_dim"]
        ).to(device)
        
        print(f"\nTesting forward pass with input shape: {history.shape}")
        
        with torch.no_grad():
            output = model(history, return_primitives=True)
        
        print(f"\nOutput shapes:")
        print(f"  Trajectories: {output['trajectories'].shape}")
        print(f"  Confidences: {output['confidences'].shape}")
        print(f"  Primitive logits: {output['primitive_logits'].shape}")
        
        print("\n✓ Model architecture test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during model architecture test: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")


def train_model_menu():
    """Train CBS model"""
    print("\n" + "-"*60)
    print("Training CBS Model")
    print("-"*60)
    
    # Check for existing checkpoint
    checkpoint_dir = config.CHECKPOINTS_DIR
    best_model_path = checkpoint_dir / "best_model.pth"
    
    resume = None
    if best_model_path.exists():
        response = input(f"\nFound existing model at {best_model_path}.\nResume training? (y/n): ").strip().lower()
        if response == 'y':
            resume = str(best_model_path)
    
    # Get training parameters
    try:
        num_epochs = input(f"\nNumber of epochs (default: {config.TRAIN_CONFIG['num_epochs']}): ").strip()
        num_epochs = int(num_epochs) if num_epochs else config.TRAIN_CONFIG['num_epochs']
    except ValueError:
        num_epochs = config.TRAIN_CONFIG['num_epochs']
    
    print(f"\nStarting training with {num_epochs} epochs...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    print(f"Logs will be saved to: {config.LOGS_DIR}")
    
    try:
        train(num_epochs=num_epochs, resume_from=resume)
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")


def generate_videos_menu():
    """Generate prediction videos"""
    print("\n" + "-"*60)
    print("Generate Prediction Videos")
    print("-"*60)
    
    # Check for model
    checkpoint_dir = config.CHECKPOINTS_DIR
    best_model_path = checkpoint_dir / "best_model.pth"
    
    model_path = None
    if best_model_path.exists():
        model_path = str(best_model_path)
        print(f"Using model: {model_path}")
    else:
        response = input("No trained model found. Use untrained model? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping video generation.")
            input("\nPress Enter to continue...")
            return
    
    try:
        num_scenarios = input("\nNumber of scenarios to visualize (default: 5): ").strip()
        num_scenarios = int(num_scenarios) if num_scenarios else 5
        
        scenario_idx = input("Starting scenario index (default: 0): ").strip()
        scenario_idx = int(scenario_idx) if scenario_idx else 0
        
        print("\nComparison Mode:")
        print("  1. CBS Only (standard visualization)")
        print("  2. CBS vs TNT - Known Scenario (similar performance)")
        print("  3. CBS vs TNT - Novel Scenario (CBS zero-shot advantage)")
        mode_choice = input("Select mode (1-3, default: 1): ").strip()
        
        mode_map = {
            '1': 'cbs_only',
            '2': 'cbs_vs_tnt_known',
            '3': 'cbs_vs_tnt_novel'
        }
        comparison_mode = mode_map.get(mode_choice, 'cbs_only')
        
        print(f"\nGenerating videos for {num_scenarios} scenarios starting from index {scenario_idx}...")
        print(f"Mode: {comparison_mode}")
        print(f"Videos will be saved to: {config.OUTPUTS_DIR}")
        
        generate_video_from_data(
            model_path=model_path,
            scenario_idx=scenario_idx,
            num_scenarios=num_scenarios,
            comparison_mode=comparison_mode
        )
        
        print("\n✓ Video generation completed!")
        
    except Exception as e:
        print(f"\n✗ Error during video generation: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")


def run_complete_pipeline():
    """Run complete pipeline: train -> evaluate -> visualize"""
    print("\n" + "-"*60)
    print("Running Complete Pipeline")
    print("-"*60)
    
    print("\nThis will:")
    print("  1. Test primitive detection")
    print("  2. Test model architecture")
    print("  3. Train the model")
    print("  4. Generate prediction videos")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Pipeline cancelled.")
        return
    
    try:
        # Step 1: Test primitive detection
        print("\n[1/4] Testing primitive detection...")
        test_primitive_detection()
        print("✓ Primitive detection OK")
        
        # Step 2: Test model
        print("\n[2/4] Testing model architecture...")
        device = config.get_device()
        model = create_model(device)
        print(f"✓ Model created ({model.count_parameters():,} parameters)")
        
        # Step 3: Train
        print("\n[3/4] Training model...")
        train()
        print("✓ Training completed")
        
        # Step 4: Generate videos
        print("\n[4/4] Generating videos...")
        best_model_path = config.CHECKPOINTS_DIR / "best_model.pth"
        if best_model_path.exists():
            generate_video_from_data(
                model_path=str(best_model_path),
                scenario_idx=0,
                num_scenarios=5
            )
            print("✓ Videos generated")
        else:
            print("⚠ No trained model found, skipping video generation")
        
        print("\n" + "="*60)
        print("✓ Complete pipeline finished successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")


def quick_demo():
    """Quick demo with interactive visualization"""
    print("\n" + "-"*60)
    print("Quick Demo - Interactive Visualization")
    print("-"*60)
    
    checkpoint_dir = config.CHECKPOINTS_DIR
    best_model_path = checkpoint_dir / "best_model.pth"
    
    model_path = None
    if best_model_path.exists():
        model_path = str(best_model_path)
        print(f"Using model: {model_path}")
    else:
        print("No trained model found. Using untrained model for demo.")
    
    try:
        run_demo(model_path=model_path)
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")


def evaluate_model_menu():
    """Evaluate trained model"""
    print("\n" + "-"*60)
    print("Evaluate Model")
    print("-"*60)
    
    checkpoint_dir = config.CHECKPOINTS_DIR
    best_model_path = checkpoint_dir / "best_model.pth"
    
    if not best_model_path.exists():
        print("No trained model found. Please train a model first.")
        input("\nPress Enter to continue...")
        return
    
    try:
        device = config.get_device()
        model = create_model(device)
        
        print(f"Loading model from {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully!")
        
        # Load validation data
        print("\nLoading validation data...")
        _, val_loader = get_data_loaders()
        
        # Evaluate
        print("\nEvaluating on validation set...")
        all_metrics = []
        
        from tqdm import tqdm
        for batch in tqdm(val_loader, desc="Evaluating"):
            history = batch['history'].to(device)
            future = batch['future'].to(device)
            
            with torch.no_grad():
                predictions = model(history, return_primitives=False)
                metrics = evaluate.evaluate_batch(predictions, future)
                all_metrics.append(metrics)
        
        # Aggregate and print
        aggregated = evaluate.aggregate_metrics(all_metrics)
        evaluate.print_metrics(aggregated, "Final Evaluation")
        
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to continue...")


def main():
    """Main execution loop"""
    print("\n" + "="*60)
    print("Welcome to CBS Trajectory Prediction System")
    print("="*60)
    
    # Print configuration
    config.print_config()
    
    while True:
        print_menu()
        
        try:
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                test_primitive_detection_menu()
            elif choice == '2':
                test_model_architecture()
            elif choice == '3':
                train_model_menu()
            elif choice == '4':
                generate_videos_menu()
            elif choice == '5':
                run_complete_pipeline()
            elif choice == '6':
                quick_demo()
            elif choice == '7':
                evaluate_model_menu()
            elif choice == '8':
                print("\nExiting CBS System. Goodbye!")
                break
            else:
                print("\nInvalid choice. Please select 1-8.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

