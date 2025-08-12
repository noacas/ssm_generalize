# Multi-GPU Implementation

This document describes the new multi-GPU implementation that allows running experiments across multiple GPUs using separate processes.

## Overview

The new implementation replaces the previous DataLoader-based approach with a true multi-process solution where:

- Each process runs on a dedicated GPU
- Seeds are distributed evenly across processes
- Each process handles `num_seeds / num_processes` seeds
- Results are collected and checkpoints are updated in real-time
- Supports 1-4 processes depending on available GPUs

## Key Features

### Process Distribution
- **1-4 processes**: Automatically determined based on available GPUs
- **Seed distribution**: Seeds are evenly distributed across processes
- **GPU assignment**: Each process is assigned to a specific GPU

### Real-time Updates
- **Progress tracking**: Each process reports progress every 10 experiments
- **Checkpoint updates**: Results are saved every 50 completed experiments
- **Error handling**: Failed experiments are logged and tracked

### Memory Management
- **GPU isolation**: Each process uses its own GPU context
- **Memory cleanup**: Automatic cleanup after each experiment
- **Cache management**: CUDA cache is cleared between experiments

## Usage

### Basic Usage
```bash
python main_multi_gpu.py --num_seeds 100 --max_gpus 4
```

### Key Parameters
- `--num_seeds`: Total number of seeds to process
- `--max_gpus`: Maximum number of GPUs to use (default: all available)
- `--checkpoint_interval`: How often to save checkpoints (in seconds)

### Example Configuration
```bash
python main_multi_gpu.py \
    --num_seeds 100 \
    --teacher_ranks 2 4 8 \
    --student_dims 4 8 16 \
    --max_gpus 2 \
    --gnc \
    --gd \
    --checkpoint_interval 1800
```

## Process Architecture

```
Main Process
├── Process 0 (GPU 0): Seeds 0-24
├── Process 1 (GPU 1): Seeds 25-49
├── Process 2 (GPU 0): Seeds 50-74
└── Process 3 (GPU 1): Seeds 75-99
```

## Communication

### Queues
- **Results Queue**: Sends experiment results from worker processes to main process
- **Checkpoint Queue**: Sends progress updates and completion signals

### Data Flow
1. Main process creates worker processes
2. Each worker process runs experiments for its assigned seeds
3. Results are sent via queue to main process
4. Main process updates progress bar and saves checkpoints
5. All processes complete and main process saves final results

## Error Handling

### Process-level Errors
- Individual experiment failures are logged but don't stop the process
- Failed experiments are reported back to main process
- Process continues with remaining experiments

### System-level Errors
- GPU memory errors trigger automatic cleanup
- Process crashes are detected and logged
- Main process continues with remaining processes

## Performance Considerations

### GPU Utilization
- Each process uses 100% of its assigned GPU
- No GPU sharing between processes
- Optimal for compute-intensive workloads

### Memory Usage
- Each process loads its own data
- Memory is freed after each experiment
- No memory sharing between processes

### Scalability
- Scales linearly with number of GPUs
- Limited by available GPU memory
- Recommended: 1-4 processes for most setups

## Testing

Run the test script to verify the implementation:
```bash
python test_multi_gpu.py
```

This will run a minimal experiment with 4 seeds across 2 GPUs.

## Monitoring

### Logs
- Each process writes to the same log file
- Process ID is included in log messages
- Progress updates every 10 experiments

### Progress Bar
- Shows total progress across all processes
- Updates in real-time as results arrive
- Displays completion percentage

### Checkpoints
- Saved every 50 experiments
- Contains all results up to that point
- Can be used to resume interrupted runs

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size or sequence length
   - Use fewer GPUs
   - Check for memory leaks

2. **Process hangs**
   - Check GPU utilization
   - Monitor system resources
   - Restart with fewer processes

3. **Slow performance**
   - Verify GPU assignment
   - Check for CPU bottlenecks
   - Monitor memory usage

### Debug Mode
Enable verbose logging by setting log level to DEBUG in the logging configuration.

## Migration from Old Implementation

The new implementation is a drop-in replacement for the old DataLoader-based approach. No changes to command-line arguments or output formats are required.

### Key Differences
- Uses multiprocessing instead of DataLoader
- Better GPU isolation
- More robust error handling
- Real-time progress updates
