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
import pandas as pd

import yaml
from datetime import datetime
from munch import Munch
import json
import itertools

ex = Experiment("t1")

# define datasets and loaders
ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True, batch_size=80,
                          num_workers=16, shuffle=True, dataset=CMD("/basedataset.get_training_set_raw"),
                          )

get_validate_loader = ex.datasets.test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                            validate=True, batch_size=80, num_workers=16,
                                            dataset=CMD("/basedataset.get_test_set_raw"))


# example Code to Attack Checkpoints:

# python3 -m experiments.dcase22.t1.attack_checkpoint --model=parall_congaterL_pretrainNone_adv3_nomix0_0lambda1_binary0_lda0_23-02231408 --gpu=2 --epochs=3



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
    lr = 1e-4
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
    da_lambda = 0.5
    num_adv_heads = 3
    lambda_scheduler = 1
    lambda_warm_up = 3
    lambda_ramp_up = 7
    random_sec = False

    # Domain Adaptation Parallel Training "par" Post training "post"
    training_method = "par"

    # In case Binary Domain Adaptation is required
    binary_da = 0

    # New Idea for Early Better Performance (Havent Been Checked Yet)
    reweight_epoch = None
    models_dir = "retrained_models"


# register extra possible configs
add_configs(ex)


@ex.command
def get_scheduler_lambda(warm_up_len=5, ramp_down_start=50, ramp_down_len=50, last_lr_value=0.1,
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

        # if not self.net.domain:
        #     self.automatic_optimization = False
        for param in self.net.parameters():
            param.requires_grad = False

        self.model_num = self.config.get("model_num", None)
        self.experiment_name = self.config.get("experiment_name", None)
        self.checkpoint_ = os.path.join("retrained_models", self.experiment_name)
        if self.config.get("experiment_name"):
            path = os.path.join(self.checkpoint_, "model.pt")
            self.net.load_state_dict(torch.load(path))

        self.dataset_domain = ["device", "location"]
        self.device_ids = self.config.device_ids
        self.label_ids = self.config.label_ids
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}
        self.high_population = {"barcelona": 0, "london": 3, "paris": 6, "prague": 7, "vienna": 9}
        self.binary_da = self.config.get("binary_da", 0)

        self.mixup_alpha = self.config.get("mixup_alpha", 0)
        self.mixstyle_p = self.config.get("mixstyle_p", 0.0)
        self.mixstyle_alpha = self.config.get("mixstyle_alpha", 0.0)
        self.mixstyle_labels = self.config.get("mixstyle_labels", False)
        self.config.model_num = self.net.model_number

        # New Early Stopping Technique
        self.reweight_epoch = self.config.get("reweight_epoch", None)
        self.acc = [] if self.reweight_epoch else None
        self.best_weights = self.state_dict() if self.reweight_epoch else None
        self.reweight_counter = 0 if self.reweight_epoch else None

        if self.mixup_alpha != 0:
            self.augment = "mixup"
        elif self.mixstyle_alpha != 0:
            self.augment = "mixstyle"
        else:
            self.augment = "nomix"
        self.num_adv_heads = self.config.get("num_adv_heads") if self.net.target_domain else 0
        self.training_domains = ["task"] + self.net.domain if self.net.target_domain else ["task"]
        self.training_method = self.config.get("training_method")
        # self.training_temp = [(0, 0), (1, 0), (0, 1)] if self.net.target_domain else [(0, 0)]
        if self.binary_da:
            self.unique_domains = [2, 2]
        else:
            self.unique_domains = [9, 12]
        self.validate_temp = self.config.get("temp_list") if self.net.target_domain else [0, 0]
        self.attacker_temperature = [0, 0]
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
        self.attackers = dict(zip(self.dataset_domain, [None] * len(self.dataset_domain)))

        for name, unique in zip(self.dataset_domain, self.unique_domains):
            self.attackers[name] = AttackerModel(self.net.embed_dim, hidden_layers=[self.net.embed_dim],
                                                 num_attributes=unique,
                                                 activation_function='ReLU').to(self.device)

    def reset_attackers(self):
        for name in self.dataset_domain:
            for layer in self.attackers[name].children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, x, temp):
        return self.net(x, temp)

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):

        x, files, y, device_indices, cities, indices = batch
        if self.binary_da:
            device_indices[device_indices > 0] = 1
            mask = (cities == 0) | (cities == 3) | (cities == 6) | (cities == 7) | (cities == 9)
            cities[~mask] = 0
            cities[mask] = 1

        tgt_domain = self.dataset_domain[optimizer_idx]
        self.attackers[tgt_domain].to(x.device)

        domain = device_indices if tgt_domain == "device" else cities
        embed_acc = {}
        embed_bacc = {}
        results = {}
        # print("Current X Size: ", x.shape)
        if self.mel:
            x_new = self.mel_forward(x)

        temp = self.attacker_temperature
        w = dict(zip(self.dataset_domain, temp)) if self.net.domain else [0, 0]
        with torch.no_grad():
            _, embed = self.forward(x_new, w)

        y_hat = self.attackers[tgt_domain](embed)
        loss = F.cross_entropy(y_hat, domain, reduction="mean")
        results[f"loss"] = loss
        results[f"{tgt_domain} loss"] = loss

        embed_acc[tgt_domain] = ((y_hat.max(dim=1)[1] == domain).float().sum() / len(domain)).item()
        embed_bacc[tgt_domain] = (balanced_accuracy_score(domain.tolist(), y_hat.max(dim=1)[1].tolist())).item()
        results[f"t_{tgt_domain} acc"] = embed_acc[tgt_domain]
        results[f"t_{tgt_domain} bacc"] = embed_bacc[tgt_domain]
        # self.log(f"t_{tgt_domain} acc:", embed_acc[tgt_domain], prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"t_{tgt_domain} bacc", embed_bacc[tgt_domain], prog_bar=True, on_epoch=False, on_step=True)
        self.log(f"w", w, prog_bar=True, on_epoch=False, on_step=True)
        return results

    def training_epoch_end(self, outputs):
        logs = {}
        # for i, tgt_domain in enumerate(self.dataset_domain):
        #     temp = [0, 0]
        #     w = dict(zip(self.dataset_domain, temp)) if self.net.domain else [0, 0]
        #     # temp = [w["device"], w["location"]] if type(w) is dict else [0, 0]

    def validation_step(self, batch, batch_idx):
        x, files, y, device_indices, cities, indices = batch
        # print("Current X Val Size: ", x.shape)
        embed_acc = {}
        embed_bacc = {}
        results = {}
        for tgt_domain in self.dataset_domain:
            domain = device_indices if tgt_domain == "device" else cities
            self.attackers[tgt_domain].to(x.device)
            if self.mel:
                x_new = self.mel_forward(x)

            temp = self.attacker_temperature
            w = dict(zip(self.dataset_domain, temp)) if self.net.domain else [0, 0]

            task_output, embed = self.forward(x_new, w)
            task_acc = ((task_output.max(dim=1)[1] == y).float().sum() / len(y))
            # print("Shape of the Embeddings:", embed.shape)
            y_hat = self.attackers[tgt_domain](embed)
            loss = F.cross_entropy(y_hat, domain, reduction="mean")
            results[f"{tgt_domain} loss"] = loss
            results[f"vtask_{tgt_domain} acc"] = task_acc

            embed_acc[tgt_domain] = ((y_hat.max(dim=1)[1] == domain).float().sum() / len(domain))
            embed_bacc[tgt_domain] = balanced_accuracy_score(domain.tolist(), y_hat.max(dim=1)[1].tolist())
            results[f"v_{tgt_domain} acc"] = embed_acc[tgt_domain]
            results[f"v_{tgt_domain} bacc"] = embed_bacc[tgt_domain]

            self.log(f"v_{tgt_domain} bacc", embed_bacc[tgt_domain].item(), prog_bar=True, on_epoch=True, on_step=False)
            self.log(f"vtask_{tgt_domain}", task_acc.item(), prog_bar=True, on_epoch=True, on_step=False)
            # self.log(f"w", w, prog_bar=True, on_epoch=True, on_step=False)

        return results

    def validation_epoch_end(self, outputs):
        logs = {}
        for tgt_domain in self.dataset_domain:
            logs[f"v_{tgt_domain} acc"] = torch.stack([x[f"v_{tgt_domain} acc"] for x in outputs]).mean()
            logs[f"v_{tgt_domain} bacc"] = torch.stack([x[f"v_{tgt_domain} acc"] for x in outputs]).mean()
            logs[f"vtask_{tgt_domain} acc"] = torch.stack([x[f"vtask_{tgt_domain} acc"] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        x, files, y, device_indices, cities, indices = batch
        # print("Current X Val Size: ", x.shape)
        embed_acc = {}
        embed_bacc = {}
        results = {}
        for tgt_domain in self.dataset_domain:
            domain = device_indices if tgt_domain == "device" else cities
            self.attackers[tgt_domain].to(x.device)
            if self.mel:
                x_new = self.mel_forward(x)

            temp = self.attacker_temperature
            w = dict(zip(self.dataset_domain, temp)) if self.net.domain else [0, 0]

            task_output, embed = self.forward(x_new, w)
            task_acc = ((task_output.max(dim=1)[1] == y).float().sum() / len(y))
            # print("Shape of the Embeddings:", embed.shape)
            y_hat = self.attackers[tgt_domain](embed)
            loss = F.cross_entropy(y_hat, domain, reduction="mean")
            results[f"{tgt_domain} loss"] = loss
            results[f"vtask_{tgt_domain} acc"] = task_acc

            embed_acc[tgt_domain] = ((y_hat.max(dim=1)[1] == domain).float().sum() / len(domain))
            embed_bacc[tgt_domain] = balanced_accuracy_score(domain.tolist(), y_hat.max(dim=1)[1].tolist())
            results[f"v_{tgt_domain} acc"] = embed_acc[tgt_domain]
            results[f"v_{tgt_domain} bacc"] = embed_bacc[tgt_domain]

            self.log(f"v_{tgt_domain} bacc", embed_bacc[tgt_domain].item(), prog_bar=True, on_epoch=True, on_step=False)
            self.log(f"vtask_{tgt_domain}", task_acc.item(), prog_bar=True, on_epoch=True, on_step=False)

        return results

    def test_epoch_end(self, outputs):
        logs = {}
        for tgt_domain in self.dataset_domain:
            logs[f"v_{tgt_domain} acc"] = torch.stack([x[f"v_{tgt_domain} acc"] for x in outputs]).mean()
            logs[f"v_{tgt_domain} bacc"] = torch.stack([x[f"v_{tgt_domain} acc"] for x in outputs]).mean()
            logs[f"vtask_{tgt_domain} acc"] = torch.stack([x[f"vtask_{tgt_domain} acc"] for x in outputs]).mean()


    def configure_optimizers(self):
        optimizer, scheduler = [], []
        for i, name in enumerate(self.dataset_domain):
            optimizer.append(get_optimizer(self.attackers[name].parameters(), lr=self.config.get("lr")))
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
def main(_run, _config, _log, _rnd, _seed):

    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()

    modul = M(ex)
    df = pd.DataFrame()
    for temp in modul.temp_list:
        trainer = ex.get_trainer()
        modul.attacker_temperature = temp
        print("Attacker Temperature:", modul.attacker_temperature)
        trainer.fit(
            modul,
            train_dataloader=train_loader,
        )
        result = trainer.test(modul, test_dataloaders=val_loader)
        print(modul.checkpoint_)
        for i, tgt_domain in enumerate(modul.dataset_domain):
            result[0][f"w {tgt_domain}"] = temp[i]
        df = df.append(result[0], ignore_index=True)
        df.to_csv(os.path.join(modul.checkpoint_, f"attack_acc_{len(modul.temp_list)}.csv"))
        modul.reset_attackers()

    # if modul.wandb:
    #     torch.save(modul.net.state_dict(), os.path.join(modul.checkpoint_, "model.pt"))
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
