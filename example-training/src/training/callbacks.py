import math
import os
from transformers import TrainerCallback


class PerplexityCallback(TrainerCallback):
    """Compute perplexity from eval_loss and print it."""
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.log_history:
            metrics = state.log_history[-1]
            if 'eval_loss' in metrics and isinstance(metrics['eval_loss'], (int, float)):
                eval_loss = metrics['eval_loss']
                try:
                    perplexity = math.exp(eval_loss)
                    print(f"eval_perplexity: {perplexity:.4f}")
                except Exception:
                    pass


class HubPushLoggerCallback(TrainerCallback):
    """
    Log a message whenever the trainer saves a checkpoint. If push_to_hub is enabled
    with hub_strategy="checkpoint", Trainer will automatically push on save; we just log it.
    """
    def on_save(self, args, state, control, **kwargs):
        ckpt = getattr(state, "best_model_checkpoint", None)
        if ckpt is None:
            ckpt = f"{args.output_dir}/checkpoint-{state.global_step}"
        pushed = "enabled" if getattr(args, "push_to_hub", False) else "disabled"
        strategy = getattr(args, "hub_strategy", None)
        repo = getattr(args, "hub_model_id", None)
        msg = (
            f"Saved checkpoint at step {state.global_step}: {ckpt}. "
            f"Hub push is {pushed}{f' (strategy={strategy}, repo={repo})' if pushed=='enabled' else ''}."
        )
        print(msg)
