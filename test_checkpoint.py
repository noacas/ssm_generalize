#!/usr/bin/env python3
"""
Test script to verify checkpoint functionality
"""

import time
import pathlib
import psutil
import threading
import signal
import sys
import os
from main_multi_gpu import CheckpointManager
from parser import parse_args

class CPUMonitor:
    """Monitor CPU usage and shutdown if it gets too high"""
    
    def __init__(self, threshold_percent=80.0, check_interval=5.0):
        self.threshold_percent = threshold_percent
        self.check_interval = check_interval
        self.running = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start CPU monitoring in a separate thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"CPU monitoring started (threshold: {self.threshold_percent}%, check interval: {self.check_interval}s)")
        
    def stop_monitoring(self):
        """Stop CPU monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                cpu_percent = psutil.cpu_percent(interval=1.0)
                if cpu_percent > self.threshold_percent:
                    print(f"WARNING: CPU usage is {cpu_percent:.1f}% (threshold: {self.threshold_percent}%)")
                    print("Shutting down program due to high CPU usage...")
                    self._graceful_shutdown()
                    break
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in CPU monitoring: {e}")
                time.sleep(self.check_interval)
                
    def _graceful_shutdown(self):
        """Gracefully shutdown the program"""
        print("Saving checkpoint before shutdown...")
        # Signal the main thread to shutdown gracefully
        os.kill(os.getpid(), signal.SIGTERM)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nReceived shutdown signal. Saving checkpoint and exiting...")
    sys.exit(0)

def test_checkpoint_manager():
    print("=== Testing Checkpoint Manager ===")
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize CPU monitor
    cpu_monitor = CPUMonitor(threshold_percent=80.0, check_interval=5.0)
    cpu_monitor.start_monitoring()
    
    try:
        # Create test args
        args = parse_args()
        args.results_dir = pathlib.Path("./test_results")
        args.checkpoint_interval = 5  # 5 seconds for testing
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(args, checkpoint_interval=5)
        
        print(f"Checkpoint directory: {checkpoint_manager.checkpoint_dir}")
        print(f"Checkpoint interval: {checkpoint_manager.checkpoint_interval} seconds")
        print(f"Teacher ranks: {args.teacher_ranks} (count: {len(args.teacher_ranks)})")
        print(f"Student dims: {args.student_dims} (count: {len(args.student_dims)})")
        print(f"Num seeds: {args.num_seeds}")
        
        # Simulate some results
        import numpy as np
        
        # Create dummy results with proper dimensions matching args
        dummy_results = {
            'gnc_gen_losses': np.random.rand(len(args.teacher_ranks), len(args.student_dims), args.num_seeds),
            'gd_gen_losses': np.random.rand(len(args.teacher_ranks), len(args.student_dims), args.num_seeds),
            'gnc_mean_priors': np.random.rand(len(args.teacher_ranks), len(args.student_dims), args.num_seeds),
            'gnc_theoretical_losses': np.random.rand(len(args.teacher_ranks), len(args.student_dims), args.num_seeds),
            'gnc_theoretical_asymptotic_losses': np.random.rand(len(args.teacher_ranks), len(args.student_dims), args.num_seeds)
        }
        
        print("\nSimulating experiment progress...")
        
        # Calculate total experiments
        total_experiments = len(args.teacher_ranks) * len(args.student_dims) * args.num_seeds
        
        # Simulate experiments completing over time
        for i in range(min(10, total_experiments)):  # Limit to 10 or total, whichever is smaller
            completed = i + 1
            total = total_experiments
            
            print(f"Completed {completed}/{total} experiments...")
            
            # Update checkpoint manager
            checkpoint_manager.update_results(dummy_results, completed, total)
            
            # Wait a bit
            time.sleep(2)
        
        # Save final checkpoint
        print("\nSaving final checkpoint...")
        checkpoint_manager.save_final_checkpoint()
        
        print("Checkpoint test completed!")
        print(f"Check checkpoints in: {checkpoint_manager.checkpoint_dir}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving checkpoint...")
        if 'checkpoint_manager' in locals():
            checkpoint_manager.save_final_checkpoint()
    except Exception as e:
        print(f"Error occurred: {e}")
        if 'checkpoint_manager' in locals():
            print("Saving checkpoint before exit...")
            checkpoint_manager.save_final_checkpoint()
    finally:
        # Stop CPU monitoring
        cpu_monitor.stop_monitoring()

if __name__ == "__main__":
    test_checkpoint_manager()
