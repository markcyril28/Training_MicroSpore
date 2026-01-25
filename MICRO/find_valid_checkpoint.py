#!/usr/bin/env python3
"""Find the latest valid (non-corrupted) checkpoint."""

import sys
from pathlib import Path
import torch


def check_checkpoint(path: Path) -> tuple[bool, str]:
    """Check if a checkpoint is valid (no NaN/Inf values).
    
    Returns:
        (is_valid, message)
    """
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        
        if 'model_state_dict' not in checkpoint:
            return False, "Missing model_state_dict"
        
        state_dict = checkpoint['model_state_dict']
        
        for name, param in state_dict.items():
            if not torch.isfinite(param).all():
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                return False, f"Non-finite in {name}: {nan_count} NaN, {inf_count} Inf"
        
        step = checkpoint.get('step', 'unknown')
        loss = checkpoint.get('loss', 'unknown')
        return True, f"Step {step}, Loss {loss:.4f}" if isinstance(loss, float) else f"Step {step}"
        
    except Exception as e:
        return False, f"Load error: {e}"


def main():
    # Find checkpoint directory
    script_dir = Path(__file__).parent
    checkpoint_dir = script_dir / "models" / "checkpoints"
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)
    
    # Get all checkpoint files, sorted by step number (descending)
    checkpoints = list(checkpoint_dir.glob("model_step_*.pt"))
    
    if not checkpoints:
        print("No checkpoints found.")
        sys.exit(1)
    
    # Sort by step number (extract from filename)
    def get_step(p: Path) -> int:
        try:
            return int(p.stem.split('_')[-1])
        except:
            return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    
    print(f"Checking {len(checkpoints)} checkpoints...\n")
    print(f"{'Checkpoint':<40} {'Status':<10} {'Details'}")
    print("-" * 80)
    
    valid_checkpoints = []
    
    for ckpt in checkpoints:
        is_valid, msg = check_checkpoint(ckpt)
        status = "✓ VALID" if is_valid else "✗ CORRUPT"
        print(f"{ckpt.name:<40} {status:<10} {msg}")
        
        if is_valid:
            valid_checkpoints.append((ckpt, get_step(ckpt), msg))
    
    print("-" * 80)
    
    if valid_checkpoints:
        best = valid_checkpoints[0]  # Already sorted descending
        print(f"\n✓ Latest valid checkpoint: {best[0].name}")
        print(f"  Path: {best[0]}")
        print(f"  Step: {best[1]}")
        print(f"\nTo resume from this checkpoint, set in train_server.sh:")
        print(f'  RESUME="{best[0]}"')
        print(f'  RESUME_LATEST=false')
    else:
        print("\n✗ No valid checkpoints found. Training will start fresh.")
    
    # Also check latest.pt
    latest_path = script_dir / "models" / "latest.pt"
    if latest_path.exists():
        is_valid, msg = check_checkpoint(latest_path)
        status = "VALID" if is_valid else "CORRUPT"
        print(f"\nlatest.pt: {status} - {msg}")


if __name__ == "__main__":
    main()
