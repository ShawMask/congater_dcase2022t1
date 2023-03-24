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
import yaml
from tqdm import tqdm
from helpers import nessi
from datetime import datetime
import json

from ba3l.experiment import Experiment
from ba3l.module import Ba3lModule

from torch.utils.data import DataLoader

from experiments.dcase22.t1.config_updates import add_configs
from helpers.utils import mixup, mixstyle
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.workersinit import worker_init_fn
from munch import Munch

ex = Experiment("t1")

# Example call with all the default config:
# python ex_esc50.py with  trainer.precision=16  -p -m mongodb_server:27000:audioset21_balanced -c "ESC50 PaSST base"
# with 2 gpus:
# DDP=2 python ex_esc50.py with  trainer.precision=16  -p -m mongodb_server:27000:audioset21_balanced -c "ESC50 PaSST base"


# define datasets and dataloaders
get_train_loader = ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True,
                                             batch_size=80, num_workers=8, shuffle=True,
                                             dataset=CMD("/basedataset.get_training_set_raw"))

get_validate_loader = ex.datasets.test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                            validate=True, batch_size=20, num_workers=8,
                                            dataset=CMD("/basedataset.get_test_set_raw"))

# prepared for evaluating fully trained model on test split of development set
ex.datasets.quantized_test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), test=True,
                                            batch_size=100, num_workers=8, dataset=CMD("/basedataset.get_test_set_raw"))

# evaluation data
get_eval_loader = ex.datasets.evaluate.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                           batch_size=10, num_workers=8, dataset=CMD("/basedataset.get_eval_set_raw"))

# define datasets and loaders
# ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True, batch_size=80,
#                           num_workers=16, shuffle=True, dataset=CMD("/basedataset.get_training_set_raw"),
#                           )

# get_validate_loader = ex.datasets.test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
#                                             validate=True, batch_size=20, num_workers=16,
#                                             dataset=CMD("/basedataset.get_test_set_raw"))


# for storing the predictions on all of development dataset
# ex.datasets.store_predictions.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), test=True,
#                           batch_size=100, num_workers=10, dataset=CMD("/basedataset.get_development_set_raw"))


# get_eval_loader = ex.datasets.evaluate.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
#                            batch_size=10, num_workers=10, dataset=CMD("/basedataset.get_eval_set_raw"))

# CUDA_VISIBLE_DEVICES=2 python -m experiments.dcase22.t1.t1_passt with dcase22_reassembled trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 models.net.s_patchout_f=6 trainer.max_epochs=250 warm_up_len=30 ramp_down_start=30 ramp_down_len=100 random_sec=1 -p -m rk2:37373:dcase22_passt_ws -c "with one-second cropping, run=1"

#python -m experiments.dcase22.t1.passt_congater with trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 models.net.s_patchout_f=6 trainer.max_epochs=250 warm_up_len=30 ramp_down_start=30 ramp_down_len=100 random_sec=1


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
    # model_num = 254
    models = {
        "net": DynamicIngredient("models.passt.congater_passt.model_ing", instance_cmd="load_congater_passt",
                                 model_number=253, num_gate_layers=1, domain='device location',
                                 n_classes=10, input_fdim=128,
                                 s_patchout_t=0, s_patchout_f=6),
        "mel": DynamicIngredient("models.passt.preprocess.model_ing",
                                 instance_cmd="AugmentMelSTFT",
                                 n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=0,
                                 timem=0,
                                 htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=1,
                                 fmax_aug_range=1)
    }
    # spectrograms are calculated on the fly
    basedataset = DynamicIngredient(
        "datasets.dcase22.dcase22t1.dataset", audio_processor=dict(sr=32000, resample_only=True))
    trainer = dict(max_epochs=750, gpus=1, weights_summary='full', benchmark=True)
    device_ids = {'a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6'}
    label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                 'street_pedestrian', 'street_traffic', 'tram']
    lr = 1e-3
    ramp_down_len = 10
    ramp_down_start = 3
    warm_up_len = 3

    weight_decay = 0.001
    # uses mixup with alpha=0.3
    # mixup_alpha = 0.0
    mixstyle_alpha = 0.4
    mixstyle_p = 0.5
    da_lambda = 0.5
    lambda_scheduler = 1
    experiment_name = 0
    temp_list = [0, 1]
    wandb = 0
    project_name = "DCASE_ConGater_test"



# register extra possible configs
add_configs(ex)


@ex.command
def get_scheduler_lambda(warm_up_len=5, ramp_down_start=50, ramp_down_len=50, last_lr_value=0.001,
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
        # self.mel = None
        # self.da_net = None
        super(M, self).__init__(experiment)

        self.device_ids = self.config.device_ids
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}
        self.mixup_alpha = self.config.get("mixup_alpha", False)
        self.mixstyle_p = self.config.get("mixstyle_p", 0.0)
        self.mixstyle_alpha = self.config.get("mixstyle_alpha", 0.1)
        self.mixstyle_labels = self.config.get("mixstyle_labels", False)
        self.label_ids = self.config.label_ids
        self.unique_domains = [9, 12]
        self.temp_list = self.config.get("temp_list")

        # in case we need embedings for the DA
        self.net.return_embed = True

        self.stored_predictions = None
        # randomly crop out 1/10 of the 10 second snippets from dcase21
        #  - dcase21 has 1/10 of the files of dcase22
        #  - multiply epochs by 10
        #  - extend learning rate schedule by factor of 10
        #  - results in exactly the same number of training steps as training on 1 second snippets of dcase22
        #  - for validation: split 10 second files into 1 second files and repeat labels 10 times
        self.random_sec = self.config.get("random_sec", False)

        now = datetime.now()
        dt_string = now.strftime("%y-%m%d%H%M")
        if self.config.get("experiment_name") == 0:
            self.model_num = self.config.get("model_num")
            self.experiment_name = f"postcongater{self.config.get('gate_activation')}{self.model_num}_" \
                                   f"{'mixup' if self.mixup_alpha else 'mixstyle'}" \
                                   f"{self.mixup_alpha if self.mixup_alpha else self.mixstyle_alpha}" \
                                   f"_{self.config.get('lambda_scheduler')}lambda{self.config.get('da_lambda')}_{dt_string}"
        else:

            self.experiment_name = self.config.get("experiment_name")
            self.config.model_num = int(self.experiment_name.split('_')[1][1:])
            self.model_num = int(self.experiment_name.split('_')[1][1:])

        self.checkpoint_ = os.path.join("retrained_models", self.experiment_name)
        try:
            os.mkdir("retrained_models")
        except:
            print(f'checkpoint path "retrained_models" already exists')
        try:
            os.mkdir(self.checkpoint_)
        except:
            if self.config.get("experiment_name")==0:
                print(f'Model path f"{self.checkpoint_}" already exists')
                print("Warning Model will be Replaced If Training is Continued")
                key = input("Press y to continue n to abort...")
                if key=='y':
                    pass
                else:
                    exit()




        with open('da_config.yml', 'r') as f:
            self.da_config = yaml.safe_load(f)

        self.use_da = self.config.get("da_lambda", False)
        self.da_config['lambda_auto_schedule'] = True if self.config.get("lambda_scheduler") else False
        self.da_config['da_lambda'] = self.config.get("da_lambda", 0.)
        self.da_config['lambda_final'] = self.config.get("da_lambda", 0.)
        self.da_config['adv_config']['da_net_config']['layers_width'] = [self.net.embed_dim]


        self.da_models = dict(zip(self.net.domain, [None] * len(self.net.domain, ))) \
            if 'none' not in self.net.domain else None

        #  Creating Adversarial Models based on the Target Attribute
        for name in self.da_models.keys():
            if name == 'device':
                self.da_config['num_domains'] = self.unique_domains[0]
            elif name == 'location':
                self.da_config['num_domains'] = self.unique_domains[1]
            print(name)
            self.da_models[name] = DABase([self.net.embed_dim], **self.da_config)

        ls = ['warm_up_len', 'ramp_down_start', 'ramp_down_len', 'last_lr_value',
              'schedule_mode', 'wandb', 'model_num', 'adamw', 'weight_decay', 'preload', 'cmd', 'models',
              'basedataset']
        config_ = Munch.toDict(self.config)
        config_dict = {}
        for i in ls:
            config_dict[i] = config_[i]

        del config_dict['basedataset']['process_func']

        with open(os.path.join(self.checkpoint_, "config.yaml"), "w") as file:
            yaml.safe_dump(config_dict, file)
        with open(os.path.join(self.checkpoint_, "da_config.yaml"), "w") as file:
            yaml.safe_dump(self.da_config, file)


        if self.config.get('wandb'):
            import wandb
            self.wandb = wandb.init(
                dir=self.checkpoint_,
                project=self.config.get('project_name'),
                name=f"{self.experiment_name}",
                config=config_dict, )
        else:
            self.wandb = None

    def forward(self, x, temp=[0], active_gates='none'):
        return self.net(x, temp, active_gates)

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # print('train', self.device)
        # REQUIRED
        # print(f"Index of Optimizer is {optimizer_idx}", self.optimizers)
        x, files, y, device_indices, cities, indices = batch
        tgt_domain = self.net.domain[optimizer_idx]
        temperature = [0] * len(self.net.domain)
        temperature[optimizer_idx] = 1
        domain_labels = device_indices.to(self.device) if tgt_domain == 'device' else cities.to(self.device)
        # print(f"Current Domain: {tgt_domain}")
        self.net.set_trainable_parameters(f"{tgt_domain}+norm")
        # for n, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(n)

        # x is of shape: batch_size, 1, 320000
        # randomly pick 1 second from 10 seconds
        samples_to_pick = x.size(2) // 10
        t_start = np.random.randint(0, x.size(2)-samples_to_pick)
        # crop one second audio
        x = x[:, :, t_start:t_start+samples_to_pick]
        x = self.mel_forward(x)
        batch_size = len(y)

        if self.mixstyle_p > 0:
            if self.mixstyle_labels:
                x, rn_indices, lam = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha, mix_labels=self.mixstyle_labels)

                y_hat, embed = self.forward(x, temperature, tgt_domain)

                samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
                                F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                            1. - lam.reshape(batch_size)))
            else:
                x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)

                y_hat, embed = self.forward(x, temperature, tgt_domain)
                samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        elif self.mixup_alpha:
            rn_indices, lam = mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

            y_hat, embed = self.forward(x, temperature, tgt_domain)

            samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
                            F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(batch_size)))
        else:
            y_hat, embed = self.forward(x, temperature, tgt_domain)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        da_loss, da_info = self.da_models[tgt_domain].get_da_loss([embed], domain_labels)
        samples_da_loss = torch.ones_like(samples_loss) * da_loss.detach()
        loss = samples_loss.mean() + da_loss
        # print(samples_loss.mean(), da_loss)
        adv_net_accuracy = da_info['embed_accuracies'][0]

        samples_loss = samples_loss.detach()
        devices = [d.rsplit("-", 1)[1][:-4] for d in files]
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred = (preds == y).sum()
        results = {f"{tgt_domain}_loss": samples_loss,
                   f"loss": loss,
                   f"{tgt_domain}_da_loss": samples_da_loss,
                   f"{tgt_domain}_adv_net_accuracy": adv_net_accuracy,
                   f"{tgt_domain}_n_correct_pred": n_correct_pred,
                   f"{tgt_domain}_n_pred": len(y),
                  }

        for d in self.device_ids:
            results[f"{tgt_domain}_devloss_" + d] = torch.as_tensor(0., device=self.device)
            results[f"{tgt_domain}_devcnt_" + d] = torch.as_tensor(0., device=self.device)

        for i, d in enumerate(devices):
            results[f"{tgt_domain}_devloss_" + d] = results[f"{tgt_domain}_devloss_" + d] + samples_loss[i]
            results[f"{tgt_domain}_devcnt_" + d] = results[f"{tgt_domain}_devcnt_" + d] + 1.

        return results

    def training_epoch_end(self, outputs):
        logs = {}
        for i, tgt_domain in enumerate(self.net.domain):
            current_da_lambda = self.da_models[tgt_domain].get_current_lambda()
            avg_loss = torch.stack([x[f"{tgt_domain}_loss"] for x in outputs[i][:-1]]).mean()
            train_acc = sum([x[f"{tgt_domain}_n_correct_pred"] for x in outputs[i][:-1]]) * 1.0 / sum(x[f"{tgt_domain}_n_pred"] for x in outputs[i])

            logs[f"{tgt_domain}_train.loss"] = avg_loss
            logs[f"{tgt_domain}_train_acc"] = train_acc
            logs["step"] = self.current_epoch
            logs[f"{tgt_domain}_da_lambda"] = current_da_lambda

            for d in self.device_ids:
                dev_loss = torch.stack([x[f"{tgt_domain}_devloss." + d] for x in outputs[i][:-1]]).sum()
                dev_cnt = torch.stack([x[f"{tgt_domain}_devcnt." + d] for x in outputs[i][:-1]]).sum()
                logs[f"{tgt_domain}_tloss_" + d] = dev_loss / dev_cnt
                logs[f"{tgt_domain}_tcnt_" + d] = dev_cnt

            if self.use_da:
                avg_da_loss = torch.stack([x[f"{tgt_domain}_da_loss"] for x in outputs[i][:-1]]).mean()
                avg_adv_net_acc = torch.stack([x[f"{tgt_domain}_adv_net_accuracy"] for x in outputs[i][:-1]]).mean()
                logs[f"{tgt_domain}_da_loss"] = avg_da_loss
                logs[f"{tgt_domain}_adv_net_acc"] = avg_adv_net_acc
        if self.config.get('wandb'):
            self.wandb.log(logs)
        else:
            self.log_dict(logs)

    def validation_step(self, batch, batch_idx):
        # print('val', self.device)
        x, files, y, device_indices, cities, indices = batch
        temp_1 = self.temp_list
        temp_2 = self.temp_list
        lg = {}
        results = {}
        results['temp'] = self.temp_list
        # x is of shape: batch_size, 1, 320000
        y = repeat(y, 'b -> (b 10)')
        x = rearrange(x, 'b c (slices t) -> (b slices) c t', slices=10)
        files = repeat(np.array(files), 'b -> (b 10)')
        device_indices = repeat(device_indices, 'b -> (b 10)')
        cities = repeat(cities, 'b -> (b 10)')
        indices = repeat(indices, 'b -> (b 10)')
        x = self.mel_forward(x)
        for t1 in temp_1:
            for t2 in temp_2:
                temperature = [t1, t2]

                y_hat, features = self.forward(x, temperature, 'device location')
                samples_loss = F.cross_entropy(y_hat, y, reduction="none")
                loss = samples_loss.mean()

                _, preds = torch.max(y_hat, dim=1)
                n_correct_pred_per_sample = (preds == y)
                n_correct_pred = n_correct_pred_per_sample.sum()
                devices = [d.rsplit("-", 1)[1][:-4] for d in files]
                results[f"temp{t1}_{t2}_loss"] = samples_loss
                results[f"temp{t1}_{t2}_n_correct_pred"] = n_correct_pred
                results[f"temp{t1}_{t2}_n_pred"] =len(y)
                lg[f"{t1}{t2}"] = loss.item()
                for d in self.device_ids:
                    results[f"temp{t1}_{t2}_devloss_" + d] = torch.as_tensor(0., device=self.device)
                    results[f"temp{t1}_{t2}_devcnt_" + d] = torch.as_tensor(0., device=self.device)
                    results[f"temp{t1}_{t2}_devn_correct_" + d] = torch.as_tensor(0., device=self.device)
                for i, d in enumerate(devices):
                    results[f"temp{t1}_{t2}_devloss_" + d] = results[f"temp{t1}_{t2}_devloss_" + d] + samples_loss[i]
                    results[f"temp{t1}_{t2}_devn_correct_" + d] = results[f"temp{t1}_{t2}_devn_correct_" + d] + n_correct_pred_per_sample[i]
                    results[f"temp{t1}_{t2}_devcnt_" + d] = results[f"temp{t1}_{t2}_devcnt_" + d] + 1

                for l in self.label_ids:
                    results[f"temp{t1}_{t2}_lblloss_" + l] = torch.as_tensor(0., device=self.device)
                    results[f"temp{t1}_{t2}_lblcnt_" + l] = torch.as_tensor(0., device=self.device)
                    results[f"temp{t1}_{t2}_lbln_correct_" + l] = torch.as_tensor(0., device=self.device)
                for i, l in enumerate(y):
                    results[f"temp{t1}_{t2}_lblloss_" + self.label_ids[l]] = \
                        results[f"temp{t1}_{t2}_lblloss_" + self.label_ids[l]] + samples_loss[i]
                    results[f"temp{t1}_{t2}_lbln_correct_" + self.label_ids[l]] = \
                        results[f"temp{t1}_{t2}_lbln_correct_" + self.label_ids[l]] + n_correct_pred_per_sample[i]
                    results[f"temp{t1}_{t2}_lblcnt_" + self.label_ids[l]] = results[f"temp{t1}_{t2}_lblcnt_" + self.label_ids[l]] + 1

        for n, l in lg.items():
            self.log(n, l, prog_bar=True, on_epoch=True, on_step=False)
        return results

    def validation_epoch_end(self, outputs):
        temp_1 = outputs[0]['temp']
        temp_2 = outputs[0]['temp']
        lg = {}
        logs = {}
        for t1 in temp_1:
            for t2 in temp_2:
                avg_loss = torch.stack([x[f"temp{t1}_{t2}_loss"] for x in outputs[:-1]]).mean()

                val_acc = sum([x[f"temp{t1}_{t2}_n_correct_pred"] for x in outputs[:-1]]) * 1.0 / sum(x[f"temp{t1}_{t2}_n_pred"] for x in outputs[:-1])
                logs[f"temp{t1}_{t2}_loss"] = avg_loss
                logs[f"temp{t1}_{t2}_val_acc"] = val_acc
                logs["step"] = self.current_epoch
                lg[f" acc w{t1}{t2}"] = np.round(val_acc.item(), 3)

                for d in self.device_ids:
                    dev_loss = torch.stack([x[f"temp{t1}_{t2}_devloss." + d] for x in outputs[:-1]]).sum()
                    dev_cnt = torch.stack([x[f"temp{t1}_{t2}_devcnt." + d] for x in outputs[:-1]]).sum()
                    dev_corrct = torch.stack([x[f"temp{t1}_{t2}_devn_correct_" + d] for x in outputs[:-1]]).sum()
                    logs[f"temp{t1}_{t2}_vloss_" + d] = dev_loss / dev_cnt
                    logs[f"temp{t1}_{t2}_vacc_" + d] = dev_corrct / dev_cnt
                    logs[f"temp{t1}_{t2}_vcnt_" + d] = dev_cnt
                    # device groups
                    logs[f"temp{t1}_{t2}_acc_" + self.device_groups[d]] = logs.get(f"temp{t1}_{t2}_acc_" + self.device_groups[d], 0.) + dev_corrct
                    logs[f"temp{t1}_{t2}_count_" + self.device_groups[d]] = logs.get(f"temp{t1}_{t2}_count_" + self.device_groups[d], 0.) + dev_cnt
                    logs[f"temp{t1}_{t2}_lloss_" + self.device_groups[d]] = logs.get(f"temp{t1}_{t2}_lloss_" + self.device_groups[d], 0.) + dev_loss

                for d in set(self.device_groups.values()):
                    logs[f"temp{t1}_{t2}_acc_" + d] = logs[f"temp{t1}_{t2}_acc_" + d] / logs[f"temp{t1}_{t2}_count_" + d]
                    logs[f"temp{t1}_{t2}_lloss_False" + d] = logs[f"temp{t1}_{t2}_lloss_" + d] / logs[f"temp{t1}_{t2}_count_" + d]

                for l in self.label_ids:
                    lbl_loss = torch.stack([x[f"temp{t1}_{t2}_lblloss_" + l] for x in outputs]).sum()
                    lbl_cnt = torch.stack([x[f"temp{t1}_{t2}_lblcnt_" + l] for x in outputs]).sum()
                    lbl_corrct = torch.stack([x[f"temp{t1}_{t2}_lbln_correct_" + l] for x in outputs]).sum()
                    logs[f"temp{t1}_{t2}_vloss_" + l] = lbl_loss / lbl_cnt
                    logs[f"temp{t1}_{t2}_vacc_" + l] = lbl_corrct / lbl_cnt
                    logs[f"temp{t1}_{t2}_vcnt_" + l] = lbl_cnt


        try:
            print("Current learning rate:", self.trainer.lr_schedulers[0]['scheduler'].get_last_lr())
        except:
            print("Validation Epoch No Learning Rate")
        print(lg)
        if self.config.get("experiment_name")==0:
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
                except:
                    out_[k] = v
            # out_ = {k: v.to(device='cpu', non_blocking=True) for k, v in logs.items()}
            df = pd.DataFrame.from_dict(out_, orient="index")
            print(df)
            df.to_csv(os.path.join(self.checkpoint_, "evaluation_results.csv"))




    # the test functionality is exclusively used to store predictions on all samples of the development set
    def test_step(self, batch, batch_idx):
        results = self.validation_step(batch, batch_idx)
        return results

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f = batch
        if self.mel:
            x = self.mel_forward(x)
        y_hat, _ = self.forward(x)
        return f, y_hat

    def configure_optimizers(self):

        if 'none' in self.net.domain:
            optimizer = get_optimizer(self.parameters())
            # torch.optim.Adam(self.parameters(), lr=self.config.lr)
            return {
                'optimizer': optimizer,
                'lr_scheduler': get_lr_scheduler(optimizer)
            }
        else:
            #  Creating Adversarial Models based on the Target Attribute With Optimizer to train them
            da_optimizers = []
            da_schedulers = []

            for name, model in self.da_models.items():
                param_groups = []
                da_parameters = model.get_da_params()
                for param_name, parameter in self.net.named_parameters():
                    if name in param_name:
                        param_groups.append(parameter)
                optimizer = get_optimizer(param_groups + da_parameters)
                da_optimizers.append(optimizer)
                da_schedulers.append(get_lr_scheduler(optimizer))
            return da_optimizers, da_schedulers

            # REQUIRED
            # can return multiple optimizers and learning_rate schedulers
            # (LBFGS it is automatically supported, no need for closure function)


    def configure_callbacks(self):
        return get_extra_checkpoint_callback()

    def load_predictions(self, path):
        self.stored_predictions = torch.load(path, map_location=torch.device("cpu"))


@ex.command
def get_extra_checkpoint_callback(save_last_n=None):
    if save_last_n is None:
        return []
    return [ModelCheckpoint(monitor="step", verbose=True, save_top_k=save_last_n, mode='max')]


# command to store predictions for a specific run_id:

# CUDA_VISIBLE_DEVICES=2 python -m experiments.dcase22.t1.t1_passt store_predictions with trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 preload.run_id=147
# CUDA_VISIBLE_DEVICES=3 python -m experiments.dcase22.t1.t1_passt store_predictions with trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 preload.run_id=37
# CUDA_VISIBLE_DEVICES=3 python -m experiments.dcase22.t1.t1_passt store_predictions with trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 preload.run_id=188

# CUDA_VISIBLE_DEVICES=4 python -m experiments.dcase22.t1.t1_passt store_predictions with dcase21_dataset trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 preload.run_id=174

# CUDA_VISIBLE_DEVICES=3 python -m experiments.dcase22.t1.t1_passt store_predictions with trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 preload.run_id=226

@ex.command
def store_predictions(_run, _config, _log, _rnd, _seed):
    trainer = ex.get_trainer()
    data_loader = ex.get_dataloaders(dict(test=True))
    net_statedict = get_net_state_dict()
    modul = M(ex)
    modul.net.load_state_dict(net_statedict)
    # store predictions
    res = trainer.test(modul, test_dataloaders=data_loader)


# CUDA_VISIBLE_DEVICES=2 python -m experiments.dcase22.t1.t1_passt eval_stored_predictions with trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 preload.run_id=137
# CUDA_VISIBLE_DEVICES=2 python -m experiments.dcase22.t1.t1_passt eval_stored_predictions with trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 preload.run_id=151

@ex.command
def eval_stored_predictions(_run, _config, _log, _rnd, _seed):
    trainer = ex.get_trainer()
    data_loader = ex.get_dataloaders(dict(test=True))
    net_statedict = get_net_state_dict()
    modul = M(ex)
    modul.net.load_state_dict(net_statedict)
    pred_file = get_pred_file()
    print(f"Load stored predictions from file {pred_file} and evaluate.")
    modul.load_predictions(pred_file)
    res = trainer.test(modul, test_dataloaders=data_loader)


@ex.command
def predict_unseen(preload, saque_cmd, _run, _config, _log, _rnd, _seed):
    import json

    # force overriding the config, not logged = not recommended
    trainer = ex.get_trainer()
    check_load_compitable()
    eval_loader = get_eval_loader()
    val_loader = get_validate_loader()
    ckpt = get_pl_ckpt()
    save_path = f"/share/rk6/home/fschmid/dcase22_predictions/{preload['db_name']}_{preload['run_id']}/"
    ckpt_save_path = save_path+"last_model/"
    print(f"\n\nWorking on {ckpt_save_path}\n")
    if os.path.exists(ckpt_save_path+'predictions.pt'):
        print(f"Exists! {ckpt_save_path+'predictions.pt'}\nSkipping\n\n")
    else:
        Path(ckpt_save_path).mkdir(parents=True, exist_ok=True)
        info = {}
        modul = M(ex)
        net_statedict = {k[4:]: v for k, v in ckpt['state_dict'].items() if k.startswith("net.")}
        modul.net.load_state_dict(net_statedict)
        modul.unseen_mode=False
        model_repr=""
        state_dict_desc="\n"
        state_dict_all=0
        with open(ckpt_save_path + 'all_config.json', 'w') as fp:
            json.dump(_config, fp, default=set_default_json_pickle)
        with open(ckpt_save_path + 'model_config.json', 'w') as fp:
            json.dump(_config['models']['net'], fp, default=set_default_json_pickle)
        with open(ckpt_save_path + "cmd.txt", "w") as text_file:
            print(f"{saque_cmd}\n", file=text_file)

        model_repr = str(modul.net)
        torch.save(modul.net.state_dict(), ckpt_save_path + "state_dict.pt")
        for k,v in modul.net.state_dict().items():
            p = v
            nonzero = p[p != 0].numel()
            total = p.numel()
            state_dict_desc += f"{k} total={total} nz={nonzero} type={p.dtype}\n"
            state_dict_all += total

        print(f"\n{state_dict_desc}\n\n")
        info['desc']=state_dict_desc
        print(f"\n\nNow testing Validation data, len{len(val_loader)}:")
        res=trainer.validate(modul, val_dataloaders=val_loader)
        info['val']=res
        print(res)
        modul.val_dataloader=None
        trainer.val_dataloaders = None
        torch.save(info,ckpt_save_path+"info.pt")
        print(f"\n\nUnseen len={len(eval_loader)}\n")
        print(f"Working on {ckpt_save_path}\n")
        predictions = trainer.predict(modul,dataloaders=eval_loader)
        all_files = [item for files,_ in predictions for item in files]
        assert all_files[-1]=="audio/118799.wav"
        assert all_files[0]=="audio/0.wav"
        all_predictions= torch.cat([torch.as_tensor(p) for _,p in predictions],0)
        torch.save(all_predictions,ckpt_save_path+"predictions.pt")
        with open(ckpt_save_path+"desc.txt", "w") as text_file:
            print(f"\n{state_dict_desc}\n\n",  file=text_file)
            print(f"Validation (part of training) results={res}", file=text_file)
            print(f"Model representation (part of training) results={model_repr}", file=text_file)

        print(f"saved predicts {all_predictions.shape} to {ckpt_save_path+'predictions.pt'}\n\n ")
        print(res)


@ex.command
def check_load_compitable(preload, _config):
    DB_URL = preload['db_server'] or "mongodb://rk2:37373/?retryWrites=true&w=majority"
    DB_NAME = preload['db_name'] or "ast_dcase22t1"
    import pymongo
    from pymongo import MongoClient
    mongodb_client = MongoClient(DB_URL)
    mongodb = mongodb_client[DB_NAME]
    e = mongodb["runs"].find_one({"_id": preload['run_id']})
    assert _config["models"]['net']["arch"]==e['config']["models"]['net']["arch"]


@ex.command(prefix="preload")
def get_pred_file(run_id=None, db_server=None, db_name=None, file_name=None):
    DB_URL = db_server or "mongodb://rk2:37373/?retryWrites=true&w=majority"
    DB_NAME = db_name or "ast_dcase22t1"
    import pymongo
    from pymongo import MongoClient
    mongodb_client = MongoClient(DB_URL)
    mongodb = mongodb_client[DB_NAME]
    e = mongodb["runs"].find_one({"_id": run_id})
    exp_name = e["experiment"]["name"]
    run_id = str(DB_NAME) + "_" + str(e['_id'])
    host_name = e['host']['hostname'].replace("rechenknecht", "rk").replace(".cp.jku.at", "")
    output_dir = "dcase22/malach_dcase22/" + e["config"]["trainer"]['default_root_dir']
    exp_path = f"/share/{host_name}/home/fschmid/deployment/{output_dir}/{exp_name}/{run_id}"
    assert os.path.isdir(exp_path)
    pred_path = f"{exp_path}/predictions"
    os.makedirs(pred_path, exist_ok=True)
    FILE_NAME = file_name or "default.pt"
    pred_file =f"{pred_path}/{FILE_NAME}"
    return pred_file

@ex.command
def evaluate(_run, _config, _log, _rnd, _seed, only_validation=True):
    # force overriding the config, not logged = not recommended
    trainer = ex.get_trainer()
    print(help(ex))
    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()
    net_statedict = get_net_state_dict()
    modul = M(ex)
    modul.net.load_state_dict(net_statedict)
    if not only_validation:
        print(f"\n\nNow testing training data, len{len(train_loader)}:")
        res = trainer.validate(modul, val_dataloaders=train_loader)
        print(res)
    modul.val_dataloader=None
    trainer.val_dataloaders = None
    print(f"\n\nValidation len={len(val_loader)}\n")
    res = trainer.validate(modul, val_dataloaders=val_loader)
    print("\n\n Validtaion:")
    print(res)


@ex.command(prefix="preload")
def get_net_state_dict():
    pl_ckpt = get_pl_ckpt()
    net_statedict = {k[4:]: v for k, v in pl_ckpt['state_dict'].items() if k.startswith("net.")}
    return net_statedict


@ex.command(prefix="preload")
def get_pl_ckpt(ckpt=None, run_id=None, db_server=None, db_name=None):
    if ckpt is None:
        DB_URL = db_server or "mongodb://rk2:37373/?retryWrites=true&w=majority"
        DB_NAME = db_name or "ast_dcase22t1"
        import pymongo
        from pymongo import MongoClient
        mongodb_client = MongoClient(DB_URL)
        mongodb = mongodb_client[DB_NAME]
        e = mongodb["runs"].find_one({"_id": run_id})
        exp_name = e["experiment"]["name"]
        run_id = str(DB_NAME) + "_" + str(e['_id'])
        host_name = e['host']['hostname'].replace("rechenknecht", "rk").replace(".cp.jku.at", "")
        output_dir = "dcase22/malach_dcase22/" + e["config"]["trainer"]['default_root_dir']
        ckpts_path = f"/share/{host_name}/home/fschmid/deployment/{output_dir}/{exp_name}/{run_id}/checkpoints/"
        assert os.path.isdir(ckpts_path)
        ckpt = ckpts_path + os.listdir(ckpts_path)[-1]
    elif run_id is not None:
        print("\n\nWARNING: ckpt is given ignoring the run_id argument.\n\n")
    pl_ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
    return pl_ckpt

# python -m experiments.dcase22.t1.passt_congater evaluate_trained with experiment_name="ConGater_253_lambda0.4_23-02040805" temp_list=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]  trainer.precision=16 basedataset.audio_processor.resample_only=True basedataset.audio_processor.sr=32000 models.net.s_patchout_f=6
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

    modul = M(ex)
    # sample = next(iter(train_loader))[0][0].unsqueeze(0)
    # sample = sample[:, :, :sample.size(2) // 10]
    # shape = modul.mel_forward(sample).size()
    # macc, n_params = nessi.get_model_size(modul.net, input_size=shape)

    # modul.validate_pretrained(val_loader, temperature=[0, 0], active_gates='device location')
    # trainer.test(modul, test_dataloaders=val_loader)
    # modul.temp_list = [0.2]
    # trainer.validate(modul, val_dataloaders=val_loader)
    modul.temp_list = [0, 1]
    trainer.fit(
        modul,
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
    )
    torch.save(modul.net.state_dict(), os.path.join(modul.checkpoint_, "model.pt"))
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


# @ex.command
# def evaluate_only(_run, _config, _log, _rnd, _seed):
#     # force overriding the config, not logged = not recommended
#     trainer = ex.get_trainer()
#     train_loader = ex.get_train_dataloaders()
#     val_loader = ex.get_val_dataloaders()
#     modul = M(ex)
#     modul.val_dataloader = None
#     trainer.val_dataloaders = None
#     print(f"\n\nValidation len={len(val_loader)}\n")
#     res = trainer.validate(modul, val_dataloaders=val_loader)
#     print("\n\n Validtaion:")
#     print(res)


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