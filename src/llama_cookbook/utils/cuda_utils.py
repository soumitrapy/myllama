import torch
import gc
import os
from unified_validation.utils.cloud_utils import get_cloud_path, upload_directory_with_gcsfs

def cuda_memory_clear():
    #del model, pmodel, optimizer, pbar, input_ids, attention_mask, labels, outputs
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    gc.collect()
    cuda_memory_summary()

def cuda_memory_summary():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024e6:.2f} GB,",
        f"Reserved: {torch.cuda.memory_reserved() / 1024e6:.2f} GB,",
        f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024e6:.2f} GB,",
        f"Usage: {torch.cuda.memory_reserved()*100 / torch.cuda.get_device_properties(0).total_memory :.2f} %"
        )

from transformers import TrainerCallback
class GCSUploadCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # args.output_dir is the checkpoint directory
        saving_path = get_cloud_path('/'.join(args.output_dir.split('/')[2:]))
        os.makedirs(saving_path, exist_ok=True)
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        saving_dir = saving_path+f"/checkpoint-{state.global_step}"
        # Example: upload to GCS after each checkpoint save
        upload_directory_with_gcsfs(
            checkpoint_dir,
            saving_dir
        )
