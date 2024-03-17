import os
from ast import literal_eval
import pickle

class IOACAS_dataset():
    def __init__(self, data_root=None):
        self.data_root = os.path.join(data_root, "IOACAS_pt1")
        with open(os.path.join(self.data_root, "midi.list"), 'r', encoding='gb2312') as fp:
            lines = fp.readlines()
            print(len(lines))

        with open(os.path.join(self.data_root, "query.list"), 'r', encoding='gb2312') as fp:
            lines2 = fp.readlines()
            print(len(lines2))

data_root = r"C:\Users\lun\OneDrive\Documents\CUHK\Academics\AIST\AIST3110\Project\Query_by_Singing_Humming\data\IOACAS_QBH_Coprus"
IOACAS_dataset(data_root=data_root)