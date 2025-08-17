import time
import threading
import logging
import pickle
from pathlib import Path

class CheckpointManager:
    """Manages automatic checkpointing of results every hour."""
    
    def __init__(self, args, checkpoint_interval=3600):  # 3600 seconds = 1 hour
        self.args = args
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_time = time.time()
        self.results_lock = threading.Lock()
        self.current_results = {
            'gnc_gen_losses': None,
            'gd_gen_losses': None,
            'gnc_mean_priors': None,
            'gnc_theoretical_losses': None,
            'gnc_theoretical_asymptotic_losses': None,
            'completed_experiments': 0,
            'total_experiments': 0
        }
        self.checkpoint_dir = args.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def update_results(self, new_results, completed_count, total_count):
        """Update current results and check if checkpoint is needed."""
        with self.results_lock:
            # Store references instead of copying arrays immediately
            self.current_results.update(new_results)
            self.current_results['completed_experiments'] = completed_count
            self.current_results['total_experiments'] = total_count
            
            current_time = time.time()
            if current_time - self.last_checkpoint_time >= self.checkpoint_interval:
                self._save_checkpoint()
                self.last_checkpoint_time = current_time
    
    def _save_checkpoint(self):
        """Save current results as a checkpoint."""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_filename = f'checkpoint_{timestamp}.pkl'
            checkpoint_path = self.checkpoint_dir / checkpoint_filename
            # save current results to checkpoint_path
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(self.current_results, f)
                
            progress = (self.current_results['completed_experiments'] / 
                        self.current_results['total_experiments'] * 100)
            logging.info(f"Checkpoint saved: {checkpoint_path} ({progress:.1f}% complete)")
            print(f"Checkpoint saved: {checkpoint_path} ({progress:.1f}% complete)")
                
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            print(f"Failed to save checkpoint: {e}")
    
    def save_final_checkpoint(self):
        """Save final checkpoint when all experiments are complete."""
        self._save_checkpoint()
    
    @staticmethod
    def load_latest_checkpoint(checkpoint_dir):
        """Load the most recent checkpoint from the checkpoint directory."""
        if not checkpoint_dir.exists():
            return None
        
        checkpoint_files = list(checkpoint_dir.glob('checkpoint_*.pkl'))
        if not checkpoint_files:
            return None
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_checkpoint = checkpoint_files[0]
        
        print(f"Found latest checkpoint: {latest_checkpoint}")
        
        with open(latest_checkpoint, 'rb') as f:
            results = pickle.load(f)
        return results

