import subprocess
import argparse

# Example: python -m experiments.dcase22.t1.attack_checkpoint --model=parall_congaterL_pretrainNone_adv3_nomix0_0lambda0.8_binary0_lda0_23-02231410 --gpu=5 --epochs=5

arg_parser = argparse.ArgumentParser(description='Attacking Checkpoint')
arg_parser.add_argument('--model', type=str, default=None)
arg_parser.add_argument('--gpu', type=str, default='cuda:0')
arg_parser.add_argument('--epochs', type=int, default=5)
arg_parser.add_argument('--temp_list', type=str, default=[1,0.8,0.6,0.4,0.2,0])

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


# # CUDA_VISIBLE_DEVICES=6 python -m experiments.dcase22.t1.attack_congater with check_=parlast_congaterL_pretrainNone_adv3_nomix0_0lambda0.5_23-02171152

cmd = f"CUDA_VISIBLE_DEVICES={args.gpu} python -m experiments.dcase22.t1.attack_congater with trainer.max_epochs={args.epochs} temp_list='{args.temp_list}'" \
      f" models.net.training_method='{training_method}' models.net.congater_loc='{gate_loc}' models.net.gate_activation='{gate_ac}'" \
      f" models.net.target_domain='device location' experiment_name={model_name}"

# cmd = f"python -m experiments.dcase22.t1.attack_congater with trainer.max_epochs={args.epochs} temp_list='{args.temp_list}'" \
#       f" models.net.training_method='{training_method}' models.net.congater_loc='{gate_loc}' models.net.gate_activation='{gate_ac}'" \
#       f" models.net.target_domain='device location' experiment_name={model_name}"

print(cmd)

subprocess.call(cmd, shell=True)

