import subprocess
import argparse
# parlast_congaterL_pretrainNone_adv3_nomix0_0lambda1_binary1_ldaFalse_23-02230831
# Example: python -m experiments.dcase22.t1.evaluate_checkpoint --model=parlast_congaterL_pretrainNone_adv3_nomix0_0lambda1_binary1_ldaFalse_23-02230831 --gpu=5

arg_parser = argparse.ArgumentParser(description='Evaluate Checkpoint')
arg_parser.add_argument('--model', type=str, default=None)
arg_parser.add_argument('--gpu', type=str, default='cuda:0')
arg_parser.add_argument('--temp_list', type=str, default=[1,0.95,0.9,0.85,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.15,0.1,0])

args = arg_parser.parse_args()
base_dir = "retrained_models"
model_name = args.model

models = {"net": {"n_classes": 10, "input_fdim": 128, "s_patchout_t": 0, "s_patchout_f": 6,
                          "target_domain": "device location"}}
names = model_name.split("_")
if "par" in names[0]:
    training_method = "par"
elif "post" in names[0]:
    training_method = "post"
if "last" in names[0]:
    gate_loc = "last"
elif "all" in names[0]:
    gate_loc = "all"
if names[1][-1] == "L":
    gate_ac = "L"
elif names[1][-1] == "T":
    gate_ac = "T"


cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} python -m experiments.dcase22.t1.congater_ex evaluate_trained with temp_list='{args.temp_list}'" \
      f" models.net.training_method='{training_method}' models.net.congater_loc='{gate_loc}' models.net.gate_activation='{gate_ac}'" \
      f" models.net.target_domain='device location' experiment_name={model_name}"

# cmd = f"python -m experiments.dcase22.t1.congater_ex evaluate_trained with temp_list='{args.temp_list}'" \
#       f" models.net.training_method='{training_method}' models.net.congater_loc='{gate_loc}' models.net.gate_activation='{gate_ac}'" \
#       f" models.net.target_domain='device location' experiment_name={model_name}"

print(cmd)

subprocess.call(cmd, shell=True)

