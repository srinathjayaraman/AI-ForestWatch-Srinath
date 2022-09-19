### This is a list of issues I've faced while running this code on the university PC's and TU Delft DHPC.
- The university PC's have an nVIDIA GeForce GT 730 as their GPU. This is not supported by CUDA anymore, so PyTorch started to fail suddenly. The error was `RuntimeError: CUDA error: no kernel image is available for execution on the device`. The solution was found [in the thread for this issue raised on GitHub](https://github.com/pytorch/pytorch/issues/31285). I installed pytorch version 1.12 and then executed `python train.py`, which worked (thankfully). The command to do this is ` pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html -U`.
 
- When resuming training from a checkpoint, include the **name** of the checkpoint file in the `resume` command. For example:
`python train.py --resume C:\Users\sjayaramannaga\PycharmProjects\AI-ForestWatch-Srinath\saved\models\Landsat8_UNet\0802_181258\checkpoint-epoch18.pth` (no need to use single or double quotes for the path).

- In `base_dataset.py`, a temporary directory is being created, used to store 'temp_image.npy' and then deleted before the file for a region is processed. But the original research was done on a Linux system. On Windows 10, I continuously faced an error - `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process: 'temp_numpy_saves\\temp_image.npy`. Turns out that Windows does not like the `mmap_mode` command that was used when a `temp_image_path` is created using `os.path.join`, since it is created in read-only mode. Deleting the `mmap_mode` parameter fixed the issue.

- When I tried to run `inference.py` for a specific district/year combination (battagram/2016 for example), the argument would never get passed to `inference.py`. This was because in the `if` condition that was checking to see if arguments were being passed through the command line, the district name and year were coded as `config.districts` and `config.years`. It should be `[args.districts]` and `[args.years]`.

- When I started using the Delft High Performance Compute Cluster (aka DHPC), I ran into a very weird `ImportError`:
```
Traceback (most recent call last):
  File "train.py", line 14, in <module>
    import model.model as module_arch
  File "/scratch/sjayaramannaga/AI-ForestWatch-Srinath/model/model.py", line 15, in <module>
    from torchvision import models
  File "/home/sjayaramannaga/.local/lib/python3.8/site-packages/torchvision/__init__.py", line 7, in <module>
    from torchvision import models
  File "/home/sjayaramannaga/.local/lib/python3.8/site-packages/torchvision/models/__init__.py", line 18, in <module>
    from . import quantization
  File "/home/sjayaramannaga/.local/lib/python3.8/site-packages/torchvision/models/quantization/__init__.py", line 3, in <module>
    from .mobilenet import *
  File "/home/sjayaramannaga/.local/lib/python3.8/site-packages/torchvision/models/quantization/mobilenet.py", line 1, in <module>
    from .mobilenetv2 import *  # noqa: F401, F403
  File "/home/sjayaramannaga/.local/lib/python3.8/site-packages/torchvision/models/quantization/mobilenetv2.py", line 6, in <module>
    from torch.ao.quantization import QuantStub, DeQuantStub
ImportError: cannot import name 'QuantStub' from 'torch.ao.quantization' (/apps/arch/2022r2/software/linux-rhel8-skylake_avx512/gcc-8.5.0/py-torch-1.10.0-uv5yokclab46j6l2ptyqrjwelp43xfoj/lib/python3.8/site-packages/torch/ao/quantization/__init__.py)
```
These are the details of the pytorch module installed:
```
[sjayaramannaga@login01 AI-ForestWatch-Srinath]$ python -m pip show torch
Name: torch
Version: 1.12.1
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /scratch/sjayaramannaga/.local/lib/python3.8/site-packages
Requires: typing-extensions
Required-by: torchvision
```
Turns out that the `ImportError` was looking at `py-torch-1.10.0`, whereas the version installed was `py-torch-1.12.1`. Installed a specific version using the following command, and that fixed the issue:
```
python -m pip install --user torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
I will keep updating this document as time goes on......
