import os
import sys
from pathlib import Path

import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from sacred.config_helpers import DynamicIngredient, CMD
from torch.nn import functional as F
import numpy as np
import pickle
from einops import repeat, rearrange
from da import DABase
from tqdm import tqdm

from ba3l.experiment import Experiment
from ba3l.module import Ba3lModule

from torch.utils.data import DataLoader

from experiments.dcase22.t1.config_updates import add_configs
from helpers.utils import mixup, mixstyle, RFN
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.workersinit import worker_init_fn
from sklearn.metrics import balanced_accuracy_score
from models.attacker import AttackerModel

import yaml
from datetime import datetime
from munch import Munch
import json
import itertools

ex = Experiment("t1")

# define datasets and loaders
ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True, batch_size=100,
                          num_workers=16, shuffle=True, dataset=CMD("/basedataset.get_training_set_raw"),
                          )

get_validate_loader = ex.datasets.test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                            validate=True, batch_size=100, num_workers=16,
                                            dataset=CMD("/basedataset.get_test_set_raw"))


# for storing the predictions on all of development dataset
# ex.datasets.store_predictions.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), test=True,
#                           batch_size=100, num_workers=10, dataset=CMD("/basedataset.get_development_set_raw"))
#
#
# get_eval_loader = ex.datasets.evaluate.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
#                            batch_size=10, num_workers=10, dataset=CMD("/basedataset.get_eval_set_raw"))

# CUDA_VISIBLE_DEVICES=0 python -m experiments.dcase22.t1.congater_ex with wandb=1 mixup_alpha=0.3
#
# CUDA_VISIBLE_DEVICES=0 python -m experiments.dcase22.t1.congater_ex with wandb=1 mixstyle_alpha=0.4 mixstyle_p=0.4
#
# CUDA_VISIBLE_DEVICES=0 python -m experiments.dcase22.t1.congater_ex with wandb=1

# CUDA_VISIBLE_DEVICES=6 python -m experiments.dcase22.t1.congater_ex with Lgate_last lambda_scheduler=0 da_lambda=1 training_method=par num_adv_heads=3 binary_da=1 wandb=1

# CUDA_VISIBLE_DEVICES=7 python -m experiments.dcase22.t1.congater_ex with Lgate_all lambda_scheduler=0 da_lambda=0.8 training_method=par wandb=1


@ex.config
def default_conf():
    cmd = " ".join(sys.argv)
    saque_cmd = os.environ.get("SAQUE_CMD", "").strip()
    saque_id = os.environ.get("SAQUE_ID", "").strip()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if os.environ.get("SLURM_ARRAY_JOB_ID", False):
        slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "").strip() + "_" + os.environ.get("SLURM_ARRAY_TASK_ID",
                                                                                               "").strip()
    process_id = os.getpid()
    models = {
        "net": DynamicIngredient("models.passt.congater.model_ing", arch="passt_s_swa_p16_128_ap476", n_classes=10,
                                 input_fdim=128, s_patchout_t=0, s_patchout_f=6),
        "mel": DynamicIngredient("models.passt.preprocess.model_ing",
                                 instance_cmd="AugmentMelSTFT",
                                 n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                                 timem=20,
                                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1,
                                 fmax_aug_range=1000)
    }
    # spectrograms are calculated on the fly
    basedataset = DynamicIngredient("datasets.dcase22.dcase22t1.dataset",
                                    time_shift=1000  # on waveform
                                    )
    trainer = dict(max_epochs=25, gpus=1, weights_summary='full', benchmark=True)
    device_ids = {'a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6'}
    label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                 'street_pedestrian', 'street_traffic', 'tram']
    lr = 1e-5
    da_lr = 1e-4
    ramp_down_len = 10
    ramp_down_start = 5
    warm_up_len = 3

    weight_decay = 0.001
    # uses mixup with alpha=0.3
    mixup_alpha = 0
    mixstyle_alpha = 0
    mixstyle_p = 0

    experiment_name = 0
    temp_list = [0, 1]
    wandb = 0
    project_name = "DCASE_ConGater_test2"
    da_type = "dann"
    label_aware_da = 0
    da_lambda = 0.5
    num_adv_heads = 3
    lambda_scheduler = 1
    lambda_warm_up = 3
    lambda_ramp_up = 7
    random_sec = 0

    # Domain Adaptation Parallel Training "par" Post training "post"
    training_method = "par"

    # In case Binary Domain Adaptation is required
    binary_da = 0

    # New Idea for Early Better Performance (Havent Been Checked Yet)
    reweight_epoch = None



# register extra possible configs
add_configs(ex)


@ex.command
def get_scheduler_lambda(warm_up_len=5, ramp_down_start=50, ramp_down_len=50, last_lr_value=0.01,
                         schedule_mode="exp_lin"):
    if schedule_mode == "exp_lin":
        return exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)
    if schedule_mode == "cos_cyc":
        return cosine_cycle(warm_up_len, ramp_down_start, last_lr_value)
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown for a lambda funtion.")


@ex.command
def get_lr_scheduler(optimizer, schedule_mode):
    if schedule_mode in {"exp_lin", "cos_cyc"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, get_scheduler_lambda())
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")


@ex.command
def get_optimizer(params, lr, adamw=True, weight_decay=0.0001):
    if adamw:
        print(f"\nUsing adamw weight_decay={weight_decay}!\n")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr)


class M(Ba3lModule):
    def __init__(self, experiment):
        self.mel = None
        self.da_net = None
        super(M, self).__init__(experiment)

        self.automatic_optimization = False

        self.device_ids = self.config.device_ids
        self.label_ids = self.config.label_ids
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}
        self.high_population = {"barcelona":0, "london":3, "paris":6, "prague":7, "vienna":9}
        self.binary_da = self.config.get("binary_da", 0)

        self.mixup_alpha = self.config.get("mixup_alpha", 0)
        self.mixstyle_p = self.config.get("mixstyle_p", 0.0)
        self.mixstyle_alpha = self.config.get("mixstyle_alpha", 0.0)
        self.mixstyle_labels = self.config.get("mixstyle_labels", False)
        self.config.model_num = self.net.model_number

        # New Early Stopping Technique
        self.reweight_epoch = self.config.get("reweight_epoch", None)
        self.acc = [] if self.reweight_epoch else None
        self.best_weights = self.state_dict()  if self.reweight_epoch else None
        self.reweight_counter = 0 if self.reweight_epoch else None


        if self.mixup_alpha != 0:
            self.augment = "mixup"
        elif self.mixstyle_alpha != 0:
            self.augment = "mixstyle"
        else:
            self.augment = "nomix"
        self.num_adv_heads = self.config.get("num_adv_heads") if self.net.target_domain else 0
        self.training_domains = ["task"]+self.net.domain if self.net.target_domain else ["task"]
        self.training_method = self.config.get("training_method")
        # self.training_temp = [(0, 0), (1, 0), (0, 1)] if self.net.target_domain else [(0, 0)]
        self.label_aware_da = self.config.get("label_aware_da")
        if self.binary_da:
            self.unique_domains = [2, 2]
        else:
            self.unique_domains = [9, 12]
        self.validate_temp = self.config.get("temp_list") if self.net.target_domain else [0]

        self.temp_list = list(itertools.product(self.validate_temp, self.validate_temp))
        # in case we need embedings for the DA
        self.net.return_embed = True


        rfn_relaxation = self.config.get("rfn_relaxation", False)
        if rfn_relaxation:
            self.rfn = RFN(rfn_relaxation)
        else:
            self.rfn = None

        self.stored_predictions = None

        self.random_sec = self.config.get("random_sec", False)

        now = datetime.now()
        dt_string = now.strftime("%y-%m%d%H%M")
        if self.config.get("experiment_name") == 0:
            self.model_num = self.config.get("model_num", None)
            self.experiment_name = f"{self.config.get('training_method')}{self.net.congater_loc}_congater" \
                                   f"{self.config.get('gate_activation') if self.net.target_domain else 'N'}_pretrain" \
                                   f"{self.model_num if self.net.target_domain else 'N'}_adv{self.num_adv_heads}_" \
                                   f"{self.augment}" \
                                   f"{self.mixup_alpha if self.mixup_alpha else self.mixstyle_alpha }" \
                                   f"_{self.config.get('lambda_scheduler')}" \
                                   f"lambda{self.config.get('da_lambda') if self.net.target_domain else '0'}_" \
                                   f"binary{self.binary_da}_lda{self.label_aware_da}_{dt_string}"
        else:
            self.experiment_name = self.config.get("experiment_name")
            try:
                self.config.model_num = int(self.experiment_name.split('_')[2][7:])
                self.model_num = int(self.experiment_name.split('_')[2][7:])
            except:
                self.config.model_num = None
                self.model_num = None

        self.checkpoint_ = os.path.join("retrained_models", self.experiment_name)

        if self.net.target_domain:
            with open('da_config.yml', 'r') as f:
                self.da_config = yaml.safe_load(f)

            self.da_config['da_type'] = self.config.get("da_type")
            self.da_config["embeds_idx"] = self.da_config["embeds_idx"] * self.num_adv_heads
            self.da_embeds = [self.net.embed_dim] * self.num_adv_heads
            self.da_config['lambda_auto_schedule'] = True if self.config.get("lambda_scheduler") else False
            if type(self.config.get("da_lambda")) is not list:
                self.da_config['da_lambda'] = self.config.get("da_lambda", 0.)
                self.da_config['lambda_final'] = self.config.get("da_lambda", 0.)
            self.da_config["lambda_pretrain_steps"] *= self.config.get("lambda_warm_up")
            self.da_config["lambda_inc_steps"] *= self.config.get("lambda_ramp_up")
            self.da_config['adv_config']['da_net_config']['layers_width'] = [self.net.embed_dim]
            self.da_models = dict(zip(self.net.domain, [None] * len(self.net.domain)))
            # Creating Adversarial Models based on the Target Attribute
            for i, name in enumerate(self.da_models.keys()):
                if type(self.config.get("da_lambda")) is list:
                    self.da_config['da_lambda'] = self.config.get("da_lambda")[i]
                    self.da_config['lambda_final'] = self.config.get("da_lambda")[i]
                self.da_config['num_domains'] = self.unique_domains[i]
                print(name, "unique labels:", self.unique_domains[i], "da_lambda:", self.da_config['da_lambda'])
                self.da_models[name] = DABase(self.da_embeds, **self.da_config)

        else:
            print("No Domain Adaptation")
            self.da_config = {}
            self.da_models = {}

        if self.config.get("wandb"):
            ls = ['warm_up_len', 'lr', 'ramp_down_start', 'ramp_down_len', 'last_lr_value',
                  'mixup_alpha', 'mixstyle_alpha', 'mixstyle_p','schedule_mode', 'wandb', 'model_num', 'adamw',
                  'weight_decay', 'da_type', "binary_da", 'da_lambda', 'num_adv_heads', "da_lr",
                  'lambda_scheduler', 'lambda_warm_up', "lambda_ramp_up", "training_method", "reweight_epoch",
                  'cmd', 'models', 'basedataset']


            config_ = Munch.toDict(self.config)
            self.config_dict = {}
            for i in ls:
                self.config_dict[str(i)] = config_[i]
            del config_["trainer"]["logger"]
            del config_["trainer"]["callbacks"]
            del self.config_dict['basedataset']['process_func']

            self.config_dict["trainer"] = config_["trainer"]
            self.config_dict["experiment_name"] = self.experiment_name
            self.config_dict["temp_list"] = self.temp_list
            self.config_dict["augment"] = self.augment
            print(self.config_dict)
            try:
                os.mkdir("retrained_models")
            except:
                print(f'checkpoint path "retrained_models" already exists')
            try:
                os.mkdir(self.checkpoint_)
            except:
                if self.config.get("experiment_name") == 0:
                    print(f'Model path f"{self.checkpoint_}" already exists')
                    print("Warning Model will be Replaced If Training is Continued")
                    key = input("Press y to continue n to abort...")
                    if key == 'y':
                        pass
                    else:
                        exit()
                else:
                    print(f"Evaluation Mode of pretrained model: {self.config.get('experiment_name')}")
            import wandb
            self.wandb = wandb.init(
                    dir=self.checkpoint_,
                    project=self.config.get('project_name'),
                    name=f"{self.experiment_name}",
                    config=self.config_dict)

            with open(os.path.join(self.checkpoint_, "config.yaml"), "w") as file:
                yaml.safe_dump(self.config_dict, file)
            with open(os.path.join(self.checkpoint_, "da_config.yaml"), "w") as file:
                yaml.safe_dump(self.da_config, file)
        else:
            self.wandb = None

        print("Experiment Configs:")
        print("Training Method:", self.training_method)
        print("ConGater Keys:", self.net.domain)
        print("Location of Congaters:", self.net.congater_loc)
        print(f"Activation Type:{'L type' if self.config.get('gate_activation')=='l' else 'T type'}")
        print(f"Domain Unique Labels: {self.unique_domains}")

        print("Training Domains", self.training_domains)
        print("Da Models", self.da_models.keys())
        if self.net.target_domain:
            print(f"adversarial config: da_type: {self.da_config['da_type']},"
                  f" scheduler:{self.da_config['lambda_auto_schedule']}, lambda:{self.da_config['da_lambda']},"
                  f" adv_net_config: {self.da_config['adv_config']['da_net_config']}")
        print("Number of Adversarial Heads", self.num_adv_heads)
        print("Validation Temperature", self.validate_temp)

    def forward(self, x, temp):
        return self.net(x, temp)

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def training_step(self, batch, batch_idx):
        # print(self.optimizers())
        # exit()
        optimizers = dict(zip(self.training_domains, [optimizer for optimizer in self.optimizers()]
        if type(self.optimizers()) is list else [self.optimizers()]))
        # print(optimizers)
        # schedulers = dict(zip(self.training_domains, [scheduler for scheduler in self.schedulers()]))

        x, files, y, device_indices, cities, indices = batch

        if self.binary_da:
            device_indices[device_indices > 0] = 1
            mask = (cities == 0) | (cities == 3) | (cities == 6) | (cities == 7) | (cities == 9)
            cities[~mask] = 0
            cities[mask] = 1
        # print(unique_mask_device_indices, count_mask_device_indices)

        if self.random_sec:
            samples_to_pick = x.size(2) // 10
            t_start = np.random.randint(0, x.size(2) - samples_to_pick)

            x = x[:, :, t_start:t_start + samples_to_pick]
        if self.mel:
            x = self.mel_forward(x)
        if self.rfn:
            x = self.rfn(x)
        batch_size = len(y)
        results = {}
        for tgt_domain in self.training_domains:
            if tgt_domain == "task":
                temp = [0, 0]
                w = dict(zip(self.net.domain, temp)) if self.net.domain else [0, 0]
                self.net.set_trainable_parameters("passt")
            else:
                if self.training_method == "par":
                    self.net.set_trainable_parameters(f"{tgt_domain}+head")
                elif self.training_method == "post":
                    self.net.set_trainable_parameters(f"{tgt_domain}")

                temp = [0 for _ in self.net.domain]
                w = dict(zip(self.net.domain, temp)) if self.net.domain else [0]
                w[tgt_domain] = 1
                temp = list(w.values())

            if self.mixstyle_p > 0:
                if self.mixstyle_labels:
                    x_new, rn_indices, lam = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha,
                                                  mix_labels=self.mixstyle_labels)

                    y_hat, embed = self.forward(x_new, w)

                    samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
                                    F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                                1. - lam.reshape(batch_size)))
                else:
                    x_new = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)

                    y_hat, embed = self.forward(x_new, w)
                    samples_loss = F.cross_entropy(y_hat, y, reduction="none")

            elif self.mixup_alpha:
                # print("mixup")
                rn_indices, lam = mixup(batch_size, self.mixup_alpha)
                lam = lam.to(x.device)
                x_new = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

                y_hat, embed = self.forward(x_new, w)

                samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
                                F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(batch_size)))
            else:
                y_hat, embed = self.forward(x, w)
                samples_loss = F.cross_entropy(y_hat, y, reduction="none")
            if tgt_domain != "task":
                # print("DA:", tgt_domain)
                domain_labels = device_indices if tgt_domain == "device" else cities
                if self.label_aware_da:
                    da_loss_ = torch.tensor(0., device=self.device)
                    da_acc_ = torch.tensor(0., device=self.device)
                    da_bacc = torch.tensor(0., device=self.device)
                    unique, count = torch.unique(y, return_counts=True)
                    # max_loc = torch.argmax(count)
                    # # print(unique, count, max_loc)
                    # mask_device_indices = device_indices[y == max_loc]
                    # unique_mask_device_indices, count_mask_device_indices = torch.unique(mask_device_indices,
                    #                                                                      return_counts=True)
                    # TODO: Balanced Domain Adaptastion Based on Same Labels in the Batch
                    total_count = 0
                    for label_num, label in enumerate(unique):
                        filtered_embed = embed[y == label]
                        filtered_domain_labels = domain_labels[y == label]
                        if len(torch.unique(filtered_domain_labels)) > 1:
                            total_count += count[label_num]
                            filtered_da_loss, filtered_da_info = self.da_models[tgt_domain].get_da_loss([filtered_embed] * self.num_adv_heads,
                                                                                      filtered_domain_labels, reduction="mean")
                            filtered_adv_net_accuracy = sum(filtered_da_info['embed_accuracies']) / len(filtered_da_info['embed_accuracies'])
                            filtered_adv_net_balanced_accuracy = torch.tensor(sum(filtered_da_info['embed_balanced_accuracies']) /
                                                                     len(filtered_da_info['embed_balanced_accuracies']),
                                                                     device=filtered_adv_net_accuracy.device)

                            da_loss_ += filtered_da_loss*count[label_num]/len(y)
                            da_acc_ += filtered_adv_net_accuracy*count[label_num]/len(y)
                            da_bacc += filtered_adv_net_balanced_accuracy*count[label_num]/len(y)
                            # print(tgt_domain, label_num, da_loss_)
                    da_loss = da_loss_*total_count/len(y)
                    adv_net_accuracy = da_acc_*total_count/len(y)
                    adv_net_balanced_accuracy = da_bacc*total_count/len(y)
                else:
                    da_loss, da_info = self.da_models[tgt_domain].get_da_loss([embed] * self.num_adv_heads,
                                                                              domain_labels, reduction="mean")
                    adv_net_accuracy = sum(da_info['embed_accuracies']) / len(da_info['embed_accuracies'])
                    adv_net_balanced_accuracy = torch.tensor(sum(da_info['embed_balanced_accuracies']) /
                                                             len(da_info['embed_balanced_accuracies']),
                                                             device=adv_net_accuracy.device)
            else:
                # print("No DA:", tgt_domain)
                da_loss = torch.zeros_like(samples_loss.mean())
                adv_net_accuracy = torch.zeros_like(samples_loss.mean().detach())
                adv_net_balanced_accuracy = torch.zeros_like(samples_loss.mean().detach())

            loss = samples_loss.mean() + da_loss
            loss.backward()
            optimizers[tgt_domain].step()
            optimizers[tgt_domain].zero_grad()

            samples_loss = samples_loss.detach()
            samples_da_loss = torch.ones_like(samples_loss) * da_loss.detach()
            devices = [d.rsplit("-", 1)[1][:-4] for d in files]
            _, preds = torch.max(y_hat, dim=1)
            n_correct_pred = (preds == y).sum()
            results[f"w{temp[0]}{temp[1]}_train_loss"] = samples_loss
            results[f"loss"] = loss
            results[f"w{temp[0]}{temp[1]}_train_da_loss"] = samples_da_loss
            results[f"w{temp[0]}{temp[1]}_train_adv_net_accuracy"] = adv_net_accuracy
            results[f"w{temp[0]}{temp[1]}_train_adv_net_balanced_accuracy"] = adv_net_balanced_accuracy
            results[f"w{temp[0]}{temp[1]}_train_n_correct_pred"] = n_correct_pred
            results[f"w{temp[0]}{temp[1]}_train_n_pred"] = len(y)
            results[f"w_{tgt_domain}"] = w
            self.log("loss", loss.item(), prog_bar=True, on_epoch=False, on_step=True)
            self.log(f"BA{temp[0]}{temp[1]}", adv_net_balanced_accuracy, prog_bar=True, on_epoch=False, on_step=True)


            for d in self.device_ids:
                results[f"w{temp[0]}{temp[1]}_train_devloss_" + d] = torch.as_tensor(0., device=self.device)
                results[f"w{temp[0]}{temp[1]}_train_devcnt_" + d] = torch.as_tensor(0., device=self.device)

            for i, d in enumerate(devices):
                results[f"w{temp[0]}{temp[1]}_train_devloss_" + d] = results[f"w{temp[0]}{temp[1]}_train_devloss_" + d] + samples_loss[i]
                results[f"w{temp[0]}{temp[1]}_train_devcnt_" + d] = results[f"w{temp[0]}{temp[1]}_train_devcnt_" + d] + 1.
        return results

    def training_epoch_end(self, outputs):
        logs = {}

        for i, tgt_domain in enumerate(self.training_domains):


            tgt_outputs = outputs

            w = tgt_outputs[0][f"w_{tgt_domain}"]
            temp = [w["device"], w["location"]] if type(w) is dict else [0, 0]

            # print(self.training_method, i, tgt_domain, len(outputs))
            # print(outputs[0].keys())
            if self.net.target_domain:
                current_da_lambda = self.da_models["device"].get_current_lambda()
            else:
                current_da_lambda = torch.tensor(0)
            avg_loss = torch.stack([x[f"w{temp[0]}{temp[1]}_train_loss"] for x in tgt_outputs[:-1]]).mean()
            train_acc = sum([x[f"w{temp[0]}{temp[1]}_train_n_correct_pred"] for x in tgt_outputs]) * 1.0 / sum(
                [x[f"w{temp[0]}{temp[1]}_train_n_pred"] for x in tgt_outputs])

            logs[f"w{temp[0]}{temp[1]}_train_loss"] = avg_loss
            logs[f"w{temp[0]}{temp[1]}_train_acc"] = train_acc
            logs["epoch"] = self.current_epoch
            logs[f"w{temp[0]}{temp[1]}_train_da_lambda"] = current_da_lambda
            logs[f"{tgt_domain}_lr"] = self.trainer.lr_schedulers[i]["scheduler"].get_last_lr()[0]

            for d in self.device_ids:
                dev_loss = torch.stack([x[f"w{temp[0]}{temp[1]}_train_devloss_" + d] for x in tgt_outputs[:-1]]).sum()
                dev_cnt = torch.stack([x[f"w{temp[0]}{temp[1]}_train_devcnt_" + d] for x in tgt_outputs[:-1]]).sum()
                logs[f"w{temp[0]}{temp[1]}_train_tloss_" + d] = dev_loss / dev_cnt
                logs[f"w{temp[0]}{temp[1]}_train_tcnt_" + d] = dev_cnt

                avg_da_loss = torch.stack([x[f"w{temp[0]}{temp[1]}_train_da_loss"] for x in tgt_outputs[:-1]]).mean()
                avg_adv_net_acc = torch.stack([x[f"w{temp[0]}{temp[1]}_train_adv_net_accuracy"] for x in tgt_outputs[:-1]]).mean()
                avg_adv_net_bacc = torch.stack(
                    [x[f"w{temp[0]}{temp[1]}_train_adv_net_balanced_accuracy"] for x in tgt_outputs[:-1]]).mean()
                logs[f"w{temp[0]}{temp[1]}_train_da_loss"] = avg_da_loss
                logs[f"w{temp[0]}{temp[1]}_train_adv_net_acc"] = avg_adv_net_acc
                logs[f"w{temp[0]}{temp[1]}_train_adv_net_bacc"] = avg_adv_net_bacc

        if self.config.get('wandb'):
            self.wandb.log(logs)
        else:
            self.log_dict(logs)

    def validation_step(self, batch, batch_idx):
        x, files, y, device_indices, cities, indices = batch
        lg = {}
        results = {}
        results['w'] = self.validate_temp

        if self.binary_da:
            device_indices[device_indices > 0] = 1
            mask = (cities == 0) | (cities == 3) | (cities == 6) | (cities == 7) | (cities == 9)
            cities[~mask] = 0
            cities[mask] = 1


        if self.random_sec:
            # x is of shape: batch_size, 1, 320000
            y = repeat(y, 'b -> (b 10)')
            x = rearrange(x, 'b c (slices t) -> (b slices) c t', slices=10)
            files = repeat(np.array(files), 'b -> (b 10)')
            device_indices = repeat(device_indices, 'b -> (b 10)')
            cities = repeat(cities, 'b -> (b 10)')
            indices = repeat(indices, 'b -> (b 10)')

        if self.mel:
            x = self.mel_forward(x)
        if self.rfn:
            x = self.rfn(x)
        for temp in self.temp_list:
            w = dict(zip(self.net.domain, temp)) if self.net.domain else [0]
            y_hat, embed = self.forward(x, w)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")
            loss = samples_loss.mean()

            _, preds = torch.max(y_hat, dim=1)
            n_correct_pred_per_sample = (preds == y)
            n_correct_pred = n_correct_pred_per_sample.sum()
            devices = [d.rsplit("-", 1)[1][:-4] for d in files]
            results[f"w{temp[0]}_{temp[1]}_val_loss"] = samples_loss
            results[f"w{temp[0]}_{temp[1]}_val_n_correct_pred"] = n_correct_pred
            results[f"w{temp[0]}_{temp[1]}_val_n_pred"] =len(y)
            lg[f"{temp[0]}{temp[1]}"] = loss.item()

            for d in self.device_ids:
                results[f"w{temp[0]}_{temp[1]}_val_devloss_" + d] = torch.as_tensor(0., device=self.device)
                results[f"w{temp[0]}_{temp[1]}_val_devcnt_" + d] = torch.as_tensor(0., device=self.device)
                results[f"w{temp[0]}_{temp[1]}_val_devn_correct_" + d] = torch.as_tensor(0., device=self.device)
            for i, d in enumerate(devices):
                results[f"w{temp[0]}_{temp[1]}_val_devloss_" + d] = results[f"w{temp[0]}_{temp[1]}_val_devloss_" + d] + samples_loss[i]
                results[f"w{temp[0]}_{temp[1]}_val_devn_correct_" + d] = results[f"w{temp[0]}_{temp[1]}_val_devn_correct_" + d] + n_correct_pred_per_sample[i]
                results[f"w{temp[0]}_{temp[1]}_val_devcnt_" + d] = results[f"w{temp[0]}_{temp[1]}_val_devcnt_" + d] + 1

            for l in self.label_ids:
                results[f"w{temp[0]}_{temp[1]}_val_lblloss_" + l] = torch.as_tensor(0., device=self.device)
                results[f"w{temp[0]}_{temp[1]}_val_lblcnt_" + l] = torch.as_tensor(0., device=self.device)
                results[f"w{temp[0]}_{temp[1]}_val_lbln_correct_" + l] = torch.as_tensor(0., device=self.device)
            for i, l in enumerate(y):
                results[f"w{temp[0]}_{temp[1]}_val_lblloss_" + self.label_ids[l]] = \
                    results[f"w{temp[0]}_{temp[1]}_val_lblloss_" + self.label_ids[l]] + samples_loss[i]
                results[f"w{temp[0]}_{temp[1]}_val_lbln_correct_" + self.label_ids[l]] = \
                    results[f"w{temp[0]}_{temp[1]}_val_lbln_correct_" + self.label_ids[l]] + n_correct_pred_per_sample[i]
                results[f"w{temp[0]}_{temp[1]}_val_lblcnt_" + self.label_ids[l]] = results[f"w{temp[0]}_{temp[1]}_val_lblcnt_" + self.label_ids[l]] + 1
        for n, l in lg.items():
            self.log(n, l, prog_bar=True, on_epoch=True, on_step=False)

        return results

    def validation_epoch_end(self, outputs):
        lg = {}
        logs = {}
        for temp in self.temp_list:
            avg_loss = torch.stack([x[f"w{temp[0]}_{temp[1]}_val_loss"] for x in outputs[:-1]]).mean()

            val_acc = sum([x[f"w{temp[0]}_{temp[1]}_val_n_correct_pred"] for x in outputs[:-1]]) * 1.0 / sum(
                x[f"w{temp[0]}_{temp[1]}_val_n_pred"] for x in outputs[:-1])
            logs[f"w{temp[0]}_{temp[1]}_val_loss"] = avg_loss
            logs[f"w{temp[0]}_{temp[1]}_val_acc"] = val_acc
            logs["epoch"] = self.current_epoch
            lg[f"w{temp[0]}{temp[1]}_acc"] = np.round(val_acc.item(), 3)
            if self.reweight_epoch:
                print("Reweightining Counter:", self.reweight_counter)
                self.acc.append(np.round(val_acc.item(), 3))
                if self.acc[-1] == max(self.acc):
                    self.reweight_counter = 0
                    self.best_weights = self.state_dict()
                else:
                    self.reweight_counter += 1
                    if self.reweight_counter > self.reweight_epoch:
                        self.reweight_counter = 0
                        for name, weight in self.state_dict().items():
                            if "passt" in name:
                                self.state_dict()[name] = (self.state_dict()[name] + self.best_weights[name]) / 2

            for d in self.device_ids:
                dev_loss = torch.stack([x[f"w{temp[0]}_{temp[1]}_val_devloss_" + d] for x in outputs[:-1]]).sum()
                dev_cnt = torch.stack([x[f"w{temp[0]}_{temp[1]}_val_devcnt_" + d] for x in outputs[:-1]]).sum()
                dev_corrct = torch.stack([x[f"w{temp[0]}_{temp[1]}_val_devn_correct_" + d] for x in outputs[:-1]]).sum()
                logs[f"w{temp[0]}_{temp[1]}_val_vloss_" + d] = dev_loss / dev_cnt
                logs[f"w{temp[0]}_{temp[1]}_val_vacc_" + d] = dev_corrct / dev_cnt
                logs[f"w{temp[0]}_{temp[1]}_val_vcnt_" + d] = dev_cnt
                # device groups
                logs[f"w{temp[0]}_{temp[1]}_val_acc_" + self.device_groups[d]] = logs.get(
                    f"w{temp[0]}_{temp[1]}_val_acc_" + self.device_groups[d], 0.) + dev_corrct
                logs[f"w{temp[0]}_{temp[1]}_val_count_" + self.device_groups[d]] = logs.get(
                    f"w{temp[0]}_{temp[1]}_val_count_" + self.device_groups[d], 0.) + dev_cnt
                logs[f"w{temp[0]}_{temp[1]}_val_lloss_" + self.device_groups[d]] = logs.get(
                    f"w{temp[0]}_{temp[1]}_val_lloss_" + self.device_groups[d], 0.) + dev_loss
            # print(temp)
            # print(logs.keys())
            # print(set(self.device_groups.values()))
            # exit()
            for d in set(self.device_groups.values()):
                logs[f"w{temp[0]}_{temp[1]}_val_acc_" + d] = logs[f"w{temp[0]}_{temp[1]}_val_acc_" + d] / logs[f"w{temp[0]}_{temp[1]}_val_count_" + d]
                logs[f"w{temp[0]}_{temp[1]}_val_lloss_False" + d] = logs[f"w{temp[0]}_{temp[1]}_val_lloss_" + d] / logs[
                    f"w{temp[0]}_{temp[1]}_val_count_" + d]

            for l in self.label_ids:
                lbl_loss = torch.stack([x[f"w{temp[0]}_{temp[1]}_val_lblloss_" + l] for x in outputs]).sum()
                lbl_cnt = torch.stack([x[f"w{temp[0]}_{temp[1]}_val_lblcnt_" + l] for x in outputs]).sum()
                lbl_corrct = torch.stack([x[f"w{temp[0]}_{temp[1]}_val_lbln_correct_" + l] for x in outputs]).sum()
                logs[f"w{temp[0]}_{temp[1]}_val_vloss_" + l] = lbl_loss / lbl_cnt
                logs[f"w{temp[0]}_{temp[1]}_val_vacc_" + l] = lbl_corrct / lbl_cnt
                logs[f"w{temp[0]}_{temp[1]}_val_vcnt_" + l] = lbl_cnt

        try:
            if self.net.domain:
                print("Current lr:", np.round(self.trainer.lr_schedulers[0]["scheduler"].get_last_lr(), 7),
                      "Current da_lr", np.round(self.trainer.lr_schedulers[1]["scheduler"].get_last_lr(), 6))
            else:
                print("Current lr:", np.round(self.trainer.lr_schedulers[0]["scheduler"].get_last_lr(), 7),)
        except:
            print("Test Epoch No Learning Rate")
        print(lg)
        if self.config.get("experiment_name") == 0:
            if self.config.get('wandb'):
                self.wandb.log(logs)
            else:
                self.log_dict(logs)
        else:
            import pandas as pd
            out_ = logs.copy()
            for k, v in out_.items():
                try:
                    out_[k] = v.to(device='cpu', non_blocking=True)
                    out_[k] = out_[k].numpy()
                except:
                    out_[k] = v
            df = pd.DataFrame.from_dict(out_, orient="index")
            df.to_csv(os.path.join(self.checkpoint_, f"evaluation_results_{len(self.temp_list)}.csv"))


    # the test functionality is exclusively used to store predictions on all samples of the development set
    def test_step(self, batch, batch_idx):
        x, files, y, device_indices, cities, indices = batch
        if self.stored_predictions is not None:
            y_hat = self.stored_predictions[indices].to(y.device)
        else:
            if self.mel:
                x = self.mel_forward(x)
            if self.rfn:
                x = self.rfn(x)
            y_hat, embed = self.forward(x)

        samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        loss = samples_loss.mean()

        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == y)
        n_correct_pred = n_correct_pred_per_sample.sum()
        results = {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y),
                   "sample_indices": indices, "logits": y_hat}
        return results

    def test_epoch_end(self, outputs):
        if self.stored_predictions is None:
            # store predictions
            sample_indices = torch.cat([x['sample_indices'] for x in outputs]).cpu()
            logits = torch.cat([x['logits'] for x in outputs]).cpu()
            _, sorted_sample_indices = torch.sort(sample_indices)
            logits_sorted = logits[sorted_sample_indices]
            pred_file = get_pred_file()
            print(f"Storing predictions for {len(sample_indices)} audio samples to: {pred_file}")
            torch.save(logits_sorted, pred_file)

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'val.loss': avg_loss, 'val_acc': val_acc}
        self.log_dict(logs)

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f = batch
        if self.mel:
            x = self.mel_forward(x)
        y_hat, _ = self.forward(x)
        return f, y_hat

    def configure_optimizers(self):
        optimizer, scheduler = [], []
        if self.net.domain:
            for i, name in enumerate(self.training_domains):
                if name == "task":
                    optimizer.append(get_optimizer(self.parameters(), lr=self.config.get("lr")))
                    scheduler.append(get_lr_scheduler(optimizer[-1]))
                else:
                    optimizer.append(get_optimizer(list(self.parameters())+self.da_models[name].get_da_params(),
                                                   lr=self.config.get("da_lr")))
                    scheduler.append(get_lr_scheduler(optimizer[-1]))
        else:
            optimizer.append(get_optimizer(self.parameters()))
            scheduler.append(get_lr_scheduler(optimizer[-1]))

        return optimizer, scheduler

    def configure_callbacks(self):
        return get_extra_checkpoint_callback()

    def load_predictions(self, path):
        self.stored_predictions = torch.load(path, map_location=torch.device("cpu"))


@ex.command
def get_extra_checkpoint_callback(save_last_n=None):
    if save_last_n is None:
        return []
    return [ModelCheckpoint(monitor="step", verbose=True, save_top_k=save_last_n, mode='max')]


@ex.command
def evaluate_trained(_run, _config, _log, _rnd, _seed):
    # force overriding the config, not logged = not recommended
    trainer = ex.get_trainer()
    val_loader = ex.get_val_dataloaders()
    modul = M(ex)
    file_path = os.path.join('retrained_models', modul.experiment_name, 'model.pt')
    new_weights = torch.load(file_path)
    modul.net.load_state_dict(new_weights)
    print(f"Evaluate Model: {modul.experiment_name} with temperatures: {modul.temp_list}")
    print(f"\n\nValidation len={len(val_loader)}\n")
    res = trainer.validate(modul, val_dataloaders=val_loader)
    print("\n\n Validtaion:")
    print(res)


@ex.command
def main(_run, _config, _log, _rnd, _seed):
    trainer = ex.get_trainer()
    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()
    if _config["training_method"]=="par":
        modul = M(ex)

        trainer.fit(
            modul,
            train_dataloader=train_loader,
            val_dataloaders=val_loader,
        )

        if modul.wandb:
            torch.save(modul.net.state_dict(), os.path.join(modul.checkpoint_, "model.pt"))

    elif _config["training_method"] == "post":
        modul = M(ex)
        modul.training_domains = ["task"]
        full_temp_list = modul.temp_list
        modul.temp_list = [full_temp_list[0]]
        trainer.fit(
            modul,
            train_dataloader=train_loader,
            val_dataloaders=val_loader,
        )

        trainer = ex.get_trainer()
        modul.training_domains = modul.net.domain
        modul.temp_list = full_temp_list[1:]
        trainer.fit(
            modul,
            train_dataloader=train_loader,
            val_dataloaders=val_loader,
        )

        if modul.wandb:
            torch.save(modul.net.state_dict(), os.path.join(modul.checkpoint_, "model.pt"))

    # else:
    #     checkpoint_ = os.path.join("retrained_models", modul.experiment_name)
    #     torch.save(modul.net.state_dict(), os.path.join(checkpoint_, "model.pt"))


    return {"done": True}


@ex.command
def model_speed_test(_run, _config, _log, _rnd, _seed, speed_test_batch_size=100):
    '''
    Test training speed of a model
    @param _run:
    @param _config:
    @param _log:
    @param _rnd:
    @param _seed:
    @param speed_test_batch_size: the batch size during the test
    @return:
    '''

    modul = M(ex)
    modul = modul.cuda()
    batch_size = speed_test_batch_size
    print(f"\nBATCH SIZE : {batch_size}\n")
    test_length = 100
    print(f"\ntest_length : {test_length}\n")

    x = torch.ones([batch_size, 1, 128, 998]).cuda()
    target = torch.ones([batch_size, 527]).cuda()
    # one passe
    net = modul.net
    # net(x)
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    # net = torch.jit.trace(net,(x,))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    print("warmup")
    import time
    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(10):
        with  torch.cuda.amp.autocast():
            y_hat, embed = net(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, target, reduction="none").mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    t2 = time.time()
    print('warmup done:', (t2 - t1))
    torch.cuda.synchronize()
    t1 = time.time()
    print("testing speed")

    for i in range(test_length):
        with  torch.cuda.amp.autocast():
            y_hat, embed = net(x)
            loss = F.binary_cross_entropy_with_logits(y_hat, target, reduction="none").mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    t2 = time.time()
    print('test done:', (t2 - t1))
    print("average speed: ", (test_length * batch_size) / (t2 - t1), " specs/second")


@ex.command
def evaluate_only(_run, _config, _log, _rnd, _seed):
    # force overriding the config, not logged = not recommended
    trainer = ex.get_trainer()
    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()
    modul = M(ex)
    modul.val_dataloader = None
    trainer.val_dataloaders = None
    print(f"\n\nValidation len={len(val_loader)}\n")
    res = trainer.validate(modul, val_dataloaders=val_loader)
    print("\n\n Validtaion:")
    print(res)


@ex.command
def test_loaders():
    '''
    get one sample from each loader for debbuging
    @return:
    '''
    for i, b in enumerate(ex.datasets.training.get_iter()):
        print(b)
        break

    for i, b in enumerate(ex.datasets.test.get_iter()):
        print(b)
        break


def set_default_json_pickle(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError



def multiprocessing_run(rank, word_size):
    print("rank ", rank, os.getpid())
    print("word_size ", word_size)
    os.environ['NODE_RANK'] = str(rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES'].split(",")[rank]
    argv = sys.argv
    if rank != 0:
        print(f"Unobserved {os.getpid()} with rank {rank}")
        argv = argv + ["-u"]  # only rank 0 is observed
    if "with" not in argv:
        argv = argv + ["with"]

    argv = argv + [f"trainer.num_nodes={word_size}", f"trainer.accelerator=ddp"]
    print(argv)

    @ex.main
    def default_command():
        return main()

    ex.run_commandline(argv)


if __name__ == '__main__':
    # set DDP=2 forks two processes to run on two GPUs
    # the environment variable "DDP" define the number of processes to fork
    # With two 2x 2080ti you can train the full model to .47 in around 24 hours
    # you may need to set NCCL_P2P_DISABLE=1
    word_size = os.environ.get("DDP", None)
    if word_size:
        import random

        word_size = int(word_size)
        print(f"\n\nDDP TRAINING WITH WORD_SIZE={word_size}\n\n")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = f"{9999 + random.randint(0, 9999)}"  # plz no collisions
        os.environ['PL_IN_DDP_SUBPROCESS'] = '1'

        for rank in range(word_size):
            pid = os.fork()
            if pid == 0:
                print("Child Forked ")
                multiprocessing_run(rank, word_size)
                exit(0)

        pid, exit_code = os.wait()
        print(pid, exit_code)
        exit(0)

print("__main__ is running pid", os.getpid(), "in module main: ", __name__)


@ex.automain
def default_command():
    return main()