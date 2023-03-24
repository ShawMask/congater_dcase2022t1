import scipy
from sacred.config_helpers import DynamicIngredient, CMD
import torch
import scipy.io.wavfile as wavfile
import os


def add_configs(ex):

    @ex.named_config
    def congater_baseline():
        models = {"net": {"n_classes": 10, "input_fdim": 128, "s_patchout_t": 0, "s_patchout_f": 6,
                          "target_domain": None,}}
        gate_activation = "N"

    @ex.named_config
    def Lgate_all_baseline():
        models = {"net": {"n_classes": 10, "input_fdim": 128, "s_patchout_t": 0, "s_patchout_f": 0,
                          "target_domain": "device location", "gate_activation": "L", "congater_loc": "all"}}
        gate_activation = "L"

    @ex.named_config
    def Lgate_all():
        models = {"net": {"n_classes": 10,"input_fdim": 128, "s_patchout_t": 0, "s_patchout_f": 6,
                          "target_domain": "device location", "gate_activation": "L", "congater_loc": "all"}}
        gate_activation = "L"
    @ex.named_config
    def Lgate_last():
        models = {"net": {"n_classes": 10,"input_fdim": 128, "s_patchout_t": 0, "s_patchout_f": 6,
                          "target_domain": "device location", "gate_activation": "L", "congater_loc": "last"}}
        gate_activation = "L"
    @ex.named_config
    def Tgate_all():
        models = {"net": {"n_classes": 10,"input_fdim": 128, "s_patchout_t": 0, "s_patchout_f": 6,
                          "target_domain": "device location", "gate_activation": "T", "congater_loc": "all"}}
        gate_activation = "T"
    @ex.named_config
    def Tgate_last():
        models = {"net": {"n_classes": 10,"input_fdim": 128, "s_patchout_t": 0, "s_patchout_f": 6,
                          "target_domain": "device location", "gate_activation": "T", "congater_loc": "last"}}
        gate_activation = "T"

    @ex.named_config
    def pretrain_256():
        models = {
            "net": DynamicIngredient("models.passt.congater.model_ing", instance_cmd="load_congater",
                                     model_number=256, dcase_weight=True, num_gate_layers=1, gate_activation="l",
                                     domain=None,
                                     n_classes=10, input_fdim=128,
                                     s_patchout_t=0, s_patchout_f=6), }

        model_num = 253
        gate_activation = "l"

    @ex.named_config
    def check_(path):
        models = {"net": {"n_classes": 10, "input_fdim": 128, "s_patchout_t": 0, "s_patchout_f": 6,
                          "target_domain": "device location"}}
        names = path.split("_")
        if "par" in names[0]:
            training_method = "par"
        elif "post" in names[0]:
            training_method = "post"
        if "last" in names[0]:
            models = {"net": {"congater_loc": "last"}}
        if names[1][-1] == "L":
            models = {"net": {"gate_activation": "L"}}
        elif names[1][-1] == "T":
            models = {"net": {"gate_activation": "T"}}
        print(models)
        exit()













