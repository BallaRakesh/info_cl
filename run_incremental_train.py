#!/usr/bin/env python3
"""
Example script for running incremental training with pre-trained model.

Usage:
    source ~/env_ml_ops/bin/activate

    # Basic usage (uses default.yaml config)
    python run_incremental_train.py

    # Override new_task_key
    python run_incremental_train.py new_task_key=itr8

    # Override multiple parameters
    python run_incremental_train.py new_task_key=itr8 training_args.stage1_epochs=15

    # Use different config
    python run_incremental_train.py --config-name=default_bert

Make sure to:
1. Update 'my_model' in configs/constants.py with your pre-trained model path
2. Add your new task data to 'manual_task_plan' in configs/constants.py
   Example:
   manual_task_plan = {
       ...
       "itr8": [
           "New Label 1",
           "New Label 2",
           "New Label 3"
       ]
   }
3. Ensure your train/test data files are properly set up
"""

import os
import sys
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
from pathlib import Path

from train.IncrementalHyperTrain import incremental_hyper_train
from configs.constants import my_model, memory_path, new_task_key

logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = "false"


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    """
    Main function for incremental training using Hydra configuration.

    Configuration is loaded from configs/default.yaml (or other yaml specified with --config-name)
    You can override any parameter from command line, e.g.:
        python run_incremental_train.py new_task_key=itr8 training_args.stage1_epochs=15
    """

    # Merge configuration sections
    args = OmegaConf.create()
    args = OmegaConf.merge(args, cfg.task_args, cfg.model_args, cfg.training_args)

    args.new_task_key = new_task_key  # Will auto-detect last key in manual_task_plan
    print(args.new_task_key)
    # Convert to SimpleNamespace for compatibility
    args = SimpleNamespace(**args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info("=" * 80)
    logger.info("Incremental Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Pre-trained model: {my_model}")
    logger.info(f"Previous memory path: {memory_path}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"New task key: {args.new_task_key or 'Auto-detect (last key in manual_task_plan)'}")
    logger.info(f"Stage 1 type: {args.stage1_type} ({args.stage1_epochs} epochs)")
    logger.info(f"Stage 2 type: {args.stage2_type} ({args.stage2_epochs} epochs)")
    logger.info(f"Memory size: {args.memory_size} exemplars per class")
    logger.info(f"Model architecture: {args.model_name_or_path}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Batch size: {args.train_batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    # Run incremental training
    results = incremental_hyper_train(
        args=args,
        data_collator=None,
        model_checkpoint_path=None,  # Will use my_model from constants
        previous_memory_path=None,    # Will use memory_path from constants
        new_task_key=args.new_task_key,
    )

    logger.info("=" * 80)
    logger.info("Incremental Training Results")
    logger.info("=" * 80)
    logger.info(f"New task performance: {results['new_task_performance']:.3f}%")
    logger.info(f"Overall performance: {results['overall_performance']:.3f}%")
    logger.info(f"Updated model saved to: {results['model_path']}")
    logger.info(f"Updated memory saved to: {results['memory_path']}")
    logger.info(f"Next task ID will be: {results['new_task_id'] + 1}")
    logger.info("=" * 80)
    logger.info(f"\nFor next incremental training, update configs/constants.py:")
    logger.info(f"  my_model = '{results['model_path']}'")
    logger.info(f"  memory_path = '{results['memory_path']}'")
    logger.info("=" * 80)



if __name__ == "__main__":
    main()
