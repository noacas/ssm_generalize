"""
Performance optimization configuration for SSM training.
This file contains optimized settings to reduce CPU bottlenecks and improve GPU utilization.
"""

import torch
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

class PerformanceConfig:
    """Configuration class for performance optimizations."""
    
    def __init__(self):
        # GPU utilization settings
        self.max_gpu_load = 0.8  # Allow higher GPU utilization
        self.max_gpu_memory = 0.8  # Allow higher memory usage
        self.max_gpus = 4  # Maximum number of GPUs to use
        
        # Multiprocessing settings
        self.max_processes = min(multiprocessing.cpu_count(), 8)  # Limit CPU processes
        
        # CUDA optimization settings
        self.cudnn_benchmark = True
        self.cudnn_deterministic = False
        self.cuda_launch_blocking = False
        
        # Memory management settings
        self.empty_cache_frequency = 10  # Clear cache every N batches
        self.pin_memory = True
        self.num_workers = 2  # For data loading if needed
        
        # Logging optimization
        self.reduced_logging = True
        self.log_tensor_samples = 5  # Number of tensor values to log
        
        # Batch processing optimization
        self.optimal_batch_size_multiplier = 2  # Increase batch sizes for better GPU utilization
        
    def apply_cuda_optimizations(self):
        """Apply CUDA optimizations to PyTorch."""
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.cuda.set_device(0)  # Set default device
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    def get_optimal_batch_size(self, base_batch_size, gpu_memory_gb=8):
        """Calculate optimal batch size based on GPU memory."""
        # Estimate memory usage per sample (rough approximation)
        memory_per_sample_mb = 50  # Adjust based on your model size
        
        # Calculate optimal batch size based on available memory
        available_memory_mb = gpu_memory_gb * 1024 * self.max_gpu_memory
        optimal_batch_size = int(available_memory_mb / memory_per_sample_mb)
        
        # Use the larger of base_batch_size or calculated optimal size
        return max(base_batch_size, optimal_batch_size)
    
    def get_process_count(self, num_experiments, num_gpus):
        """Calculate optimal number of processes."""
        # Don't create more processes than experiments or available GPUs
        return min(self.max_processes, num_experiments, num_gpus)
    
    def optimize_for_training(self, model, dataset_size):
        """Apply training-specific optimizations."""
        # Set model to training mode
        model.train()
        
        # Enable gradient computation optimization
        torch.set_grad_enabled(True)
        
        # Optimize memory allocation
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    def get_memory_stats(self, device):
        """Get current GPU memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': reserved - allocated
            }
        return None

# Global performance configuration instance
perf_config = PerformanceConfig()

def optimize_environment():
    """Apply all performance optimizations."""
    perf_config.apply_cuda_optimizations()
    
    # Set environment variables for better performance
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    print("Performance optimizations applied:")
    print(f"  CUDA benchmark: {perf_config.cudnn_benchmark}")
    print(f"  Max GPU load: {perf_config.max_gpu_load}")
    print(f"  Max GPU memory: {perf_config.max_gpu_memory}")
    print(f"  Max processes: {perf_config.max_processes}")
