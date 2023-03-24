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

from ba3l.experiment import Experiment
from ba3l.module import Ba3lModule

from torch.utils.data import DataLoader

from experiments.dcase22.t1.config_updates import add_configs
from helpers.utils import mixup, mixstyle, RFN
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.workersinit import worker_init_fn

ex = Experiment("t1")

# Example call with all the default config:
# python ex_esc50.py with  trainer.precision=16  -p -m mongodb_server:27000:audioset21_balanced -c "ESC50 PaSST base"
# with 2 gpus:
# DDP=2 python ex_esc50.py with  trainer.precision=16  -p -m mongodb_server:27000:audioset21_balanced -c "ESC50 PaSST base"

# define datasets and loaders
ex.datasets.training.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), train=True, batch_size=16,
                          num_workers=16, shuffle=True, dataset=CMD("/basedataset.get_training_set_raw"),
                          )

get_validate_loader = ex.datasets.test.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                                            validate=True, batch_size=16, num_workers=16,
                                            dataset=CMD("/basedataset.get_test_set_raw"))


# for storing the predictions on all of development dataset
ex.datasets.store_predictions.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn), test=True,
                          batch_size=100, num_workers=10, dataset=CMD("/basedataset.get_development_set_raw"))


get_eval_loader = ex.datasets.evaluate.iter(DataLoader, static_args=dict(worker_init_fn=worker_init_fn),
                           batch_size=10, num_workers=10, dataset=CMD("/basedataset.get_eval_set_raw"))


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
        "net": DynamicIngredient("models.passt.passt.model_ing", arch="passt_s_swa_p16_128_ap476", n_classes=10, input_fdim=128,
                                 s_patchout_t=0, s_patchout_f=6),
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
    lr = 0.00001
    ramp_down_len = 10
    ramp_down_start = 3
    warm_up_len = 3

    weight_decay = 0.001
    # uses mixup with alpha=0.3
    mixup_alpha = 0.3

    da_config = {
        'da_lambda': 0.0,
        'da_type': 'dann',
        'embeds_idx': (-1,),
        'da_spec_config': {
            'cmd': {
                "n_moments": 5
            }
        },
        'adv_config': {
            "da_optimizer_config": {
                "lr": 0.0001,
                "weight_decay": 0.001
            },
            "da_net_config": {
                "layers_width": [
                    1024,
                    1024,
                ],
                "act_function": "relu",
                "dropout": 0.0
            }
        }
    }

    da_domains = "all_devices"


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

        self.device_ids = self.config.device_ids
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}
        self.mixup_alpha = self.config.get("mixup_alpha", False)
        self.mixstyle_p = self.config.get("mixstyle_p", 0.0)
        self.mixstyle_alpha = self.config.get("mixstyle_alpha", 0.1)
        self.mixstyle_labels = self.config.get("mixstyle_labels", False)

        # in case we need embedings for the DA
        self.net.return_embed = True
        rfn_relaxation = self.config.get("rfn_relaxation", False)
        if rfn_relaxation:
            self.rfn = RFN(rfn_relaxation)
        else:
            self.rfn = None

        self.stored_predictions = None
        # randomly crop out 1/10 of the 10 second snippets from dcase21
        #  - dcase21 has 1/10 of the files of dcase22
        #  - multiply epochs by 10
        #  - extend learning rate schedule by factor of 10
        #  - results in exactly the same number of training steps as training on 1 second snippets of dcase22
        #  - for validation: split 10 second files into 1 second files and repeat labels 10 times
        self.random_sec = self.config.get("random_sec", False)

        self.da_config = self.config['da_config']
        self.use_da = self.da_config.get("da_lambda", False)
        if self.use_da:
            embeds_sizes = [768]  # embed dim of transformer - better: infer automatically
            self.embeds_idx = self.da_config.get("embeds_idx", (-1,))
            self.da_domains = self.config.get("da_domains", "all_devices")
            if self.da_domains == "all_devices":
                num_domains = 6
            elif self.da_domains == "real_sim" or self.da_domains == "a_rest":
                num_domains = 2
            else:
                raise NotImplementedError("No such specification for domains implemented: {}".format(self.da_domains))

    def forward(self, x):
        return self.net(x)

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, files, y, device_indices, cities, indices = batch
        if self.random_sec:
            # x is of shape: batch_size, 1, 320000
            # randomly pick 1 second from 10 seconds
            samples_to_pick = x.size(2) // 10
            t_start = np.random.randint(0, x.size(2)-samples_to_pick)
            # crop one second audio
            x = x[:, :, t_start:t_start+samples_to_pick]
        if self.mel:
            x = self.mel_forward(x)
        if self.rfn:
            x = self.rfn(x)
        batch_size = len(y)

        if self.mixstyle_p > 0:
            if self.mixstyle_labels:
                x, rn_indices, lam = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha,
                                              mix_labels=self.mixstyle_labels)

                y_hat, embed = self.forward(x)

                samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
                                F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                            1. - lam.reshape(batch_size)))
            else:
                x = mixstyle(x, self.mixstyle_p, self.mixstyle_alpha)

                y_hat, embed = self.forward(x)
                samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        elif self.mixup_alpha:
            rn_indices, lam = mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

            y_hat, embed = self.forward(x)

            samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
                            F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(batch_size)))
        else:
            y_hat, embed = self.forward(x)
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")

        da_loss = torch.as_tensor(0.).to(y_hat.device)
        adv_net_accuracy = torch.as_tensor(0.)

        loss = samples_loss.mean() + da_loss
        samples_loss = samples_loss.detach()
        devices = [d.rsplit("-", 1)[1][:-4] for d in files]
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred = (preds == y).sum()
        results = {"loss": loss, "da_loss": da_loss.cpu(), "adv_net_accuracy": adv_net_accuracy,
                   "n_correct_pred": n_correct_pred, "n_pred": len(y)}

        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)

        for i, d in enumerate(devices):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devcnt." + d] = results["devcnt." + d] + 1.

        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        train_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)

        logs = {'train.loss': avg_loss, 'train_acc': train_acc, 'step': self.current_epoch}

        for d in self.device_ids:
            dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
            dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
            logs["tloss." + d] = dev_loss / dev_cnt
            logs["tcnt." + d] = dev_cnt

        if self.use_da:
            avg_da_loss = torch.stack([x['da_loss'] for x in outputs]).mean()
            avg_adv_net_acc = torch.stack([x['adv_net_accuracy'] for x in outputs]).mean()
            logs["da_loss"] = avg_da_loss
            logs["adv_net_acc"] = avg_adv_net_acc
        self.log_dict(logs)

    def validation_step(self, batch, batch_idx):
        x, files, y, device_indices, cities, indices = batch
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
        y_hat, embed = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, y, reduction="none")
        loss = samples_loss.mean()

        self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == y)
        n_correct_pred = n_correct_pred_per_sample.sum()
        devices = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(y)}
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(devices):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_pred_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'val.loss': avg_loss, 'val_acc': val_acc, 'step': self.current_epoch}

        for d in self.device_ids:
            dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
            dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
            dev_corrct = torch.stack([x["devn_correct." + d] for x in outputs]).sum()
            logs["vloss." + d] = dev_loss / dev_cnt
            logs["vacc." + d] = dev_corrct / dev_cnt
            logs["vcnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss.False" + d] = logs["lloss." + d] / logs["count." + d]

        self.log_dict(logs)

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
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = get_optimizer(self.parameters())
        # torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': get_lr_scheduler(optimizer)
        }

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
def predict_unseen(preload,saque_cmd,_run, _config, _log, _rnd, _seed):
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


@ex.command
def main(_run, _config, _log, _rnd, _seed):
    trainer = ex.get_trainer()
    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()

    modul = M(ex)

    trainer.fit(
        modul,
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
    )

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