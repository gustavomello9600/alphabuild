import inspect
from monai.networks.nets import SwinUNETR

with open("monai_sig.txt", "w") as f:
    f.write(str(inspect.signature(SwinUNETR.__init__)))
