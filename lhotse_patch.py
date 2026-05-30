"""
Monkey-patch for lhotse library to fix Windows incompatibility.

This patches the CutSampler.__init__ method to avoid passing kwargs to object.__init__(),
which causes: TypeError: object.__init__() takes exactly one argument (the instance to initialize)

Must be imported BEFORE nemo.collections.asr.
"""

import sys

def apply_lhotse_patch():
    """Apply the monkey-patch to lhotse.dataset.sampling.base.CutSampler"""
    try:
        from lhotse.dataset.sampling.base import CutSampler
        from torch.utils.data import Sampler

        # Store original __init__
        _original_init = CutSampler.__init__

        def patched_init(
            self,
            shuffle=False,
            drop_last=False,
            world_size=None,
            rank=None,
            seed=0,
        ):
            """Patched __init__ that doesn't pass data_source to parent."""
            # Call Sampler.__init__ WITHOUT any arguments (to avoid reaching object.__init__)
            # The data_source parameter is not actually used by PyTorch's Sampler
            Sampler.__init__(self)

            # Now set all the CutSampler-specific attributes
            self.drop_last = drop_last
            self.shuffle = shuffle
            self.seed = seed
            self.epoch = 0

            from lhotse.dataset.sampling.base import SamplingDiagnostics
            self._diagnostics = SamplingDiagnostics()
            self._just_restored_state = False

            self._maybe_init_distributed(world_size=world_size, rank=rank)

            # Import filter function
            from lhotse.dataset.sampling.base import _filter_nothing
            self._filter_fn = _filter_nothing()
            self._transforms = []

        # Apply the patch
        CutSampler.__init__ = patched_init
        print("[LHOTSE_PATCH] Successfully patched CutSampler.__init__ for Windows compatibility")

    except Exception as e:
        print(f"[LHOTSE_PATCH] Warning: Failed to apply patch: {e}")
        import traceback
        traceback.print_exc()

# Auto-apply when imported
apply_lhotse_patch()
