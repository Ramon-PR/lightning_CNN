from Databases_RIR import ZeaDataset
from torch.utils.data import ConcatDataset

root = r"C:\Users\keris\Desktop\Postdoc"
a = ZeaDataset(root)

print("-------------------")

Balder = ZeaDataset(root, ["BalderRIR.mat"])
Munin = ZeaDataset(root, ["MuninRIR.mat"])
Freja = ZeaDataset(root, ["FrejaRIR.mat"])

print(Balder.__len__())
print(Munin.__len__())

BM = ConcatDataset([Balder, Munin])
print(BM.__len__())



# %%

from dataset import RirDataModule


dm = RirDataModule()

# %%

dm.setup(stage="fit")
A = dm.train_dataloader()

# %%

A

