from gln.test.model_inference import RetroGLN
from gln.common.cmd_args import cmd_args
from torch import nn
import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for test only')
cmd_opt.add_argument('-model_for_test', default=None, help='model for test')
local_args, _ = cmd_opt.parse_known_args()

class GLN_wrapper(nn.Module):
    def __init__(self, dropbox, model_dump, beam_size=50, topk=50):
        super(GLN_wrapper, self).__init__()
        self.model = RetroGLN(dropbox, model_dump)
        self.beam_size = beam_size
        self.topk = topk

    def forward(self, raw_prod, rxn_type):
        pred_struct = self.model.run(raw_prod, self.beam_size, self.topk, rxn_type=rxn_type)
        return pred_struct

raw_product = "[CH3:1][C:2]([CH3:3])([CH3:4])[O:5][C:6](=[O:7])[n:15]1[c:14]2[cH:13][cH:12][c:11]([C:9]([CH3:8])=[O:10])[cH:19][c:18]2[cH:17][cH:16]1"
rxn_type = "UNK"

model = GLN_wrapper(cmd_args.dropbox, local_args.model_for_test)
print(model(raw_product, rxn_type))