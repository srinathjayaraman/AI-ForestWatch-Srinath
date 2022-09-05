### This is a list of issues I've faced while running this code on the university PC's.
- The university PC's have an nVIDIA GeForce GT 730 as their GPU. This is not supported by CUDA anymore, so PyTorch started to fail suddenly. The error was `RuntimeError: CUDA error: no kernel image is available for execution on the device`. The solution was found [in the thread for this issue raised on GitHub](https://github.com/pytorch/pytorch/issues/31285). I installed pytorch version 1.12 and then executed `python train.py`, which worked (thankfully). The command to do this is ` pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html -U`.
 
- When resuming training from a checkpoint, include the **name** of the checkpoint file in the `resume` command. For example:
`python train.py --resume C:\Users\sjayaramannaga\PycharmProjects\AI-ForestWatch-Srinath\saved\models\Landsat8_UNet\0802_181258\checkpoint-epoch18.pth` (no need to use single or double quotes for the path).

- In `base_dataset.py`, a temporary directory is being created, used to store 'temp_image.npy' and then deleted before the file for a region is processed. But the original research was done on a Linux system. On Windows 10, I continuously faced an error - `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'temp_numpy_saves\\temp_image.npy`. Turns out that Windows does not like the `mmap_mode` command that was used when a `temp_image_path` is created using `os.path.join`, since it is created in read-only mode. Deleting the `mmap_mode` parameter fixed the issue.

I will keep updating this document as time goes on......
