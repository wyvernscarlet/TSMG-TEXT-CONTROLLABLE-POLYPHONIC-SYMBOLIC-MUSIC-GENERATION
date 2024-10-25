import pickle
import random
from argparse import ArgumentParser
from datetime import datetime
from os.path import join
from pathlib import Path
from typing import Optional
import pretty_midi as pm

import json
import numpy as np
import torch
from omegaconf import OmegaConf

from data.datasample import DataSample
from data.dataset import DataSampleNpz
from data.dataset_musicalion import DataSampleNpz_Musicalion
from data.midi_to_data import get_data_for_single_midi
from dirs import *
from lightning_learner import LightningLearner
from models.model_sdf import Polyffusion_SDF
from polydis_aftertouch import PolydisAftertouch
from sampler_ddim import DDIMSampler
from sampler_sdf import SDFSampler
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.model.unet import UNetModel
from stable_diffusion.sampler import DiffusionSampler
from utils import (
    chd_to_midi_file,
    convert_json_to_yaml,
    estx_to_midi_file,
    get_blurry_image,
    load_pretrained_chd_enc_dec,
    load_pretrained_pnotree_enc_dec,
    load_pretrained_txt_enc,
    load_pretrained_text_enc,
    prmat2c_to_midi_file,
    prmat2c_to_prmat,
    prmat_to_midi_file,
    show_image,
)
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
import torch
import muspy
import clip
from openai import OpenAI
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = ArgumentParser(description="inference a Polyffusion model")


client = OpenAI(api_key='')

ds = load_dataset("amaai-lab/MidiCaps")
train_ds = ds['train']
captionset = train_ds['caption']
midiset = train_ds['location']
key_set = train_ds['key']
ts_set = train_ds['time_signature']
tempo_set = train_ds['tempo']

test_number = 200

def dummy_cond_input(length, params):
    h = params.img_h
    w = params.img_w
    prmat2c = torch.zeros([length, 2, h, w]).to(device)
    pnotree = torch.zeros([length, h, 20, 6]).to(device)
    if "chord" in params.cond_type:
        chord = torch.zeros([length, params.chd_n_step, params.chd_input_dim]).to(
            device
        )
    else:
        chord = None
    prmat = torch.zeros([length, h, w]).to(device)
    text = None
    return prmat2c, pnotree, prmat, text


def get_data_preprocessed(song, data_type):
    prmat2c, pnotree, chord, prmat = song.get_whole_song_data()
    prmat2c_np = prmat2c.cpu().numpy()
    pnotree_np = pnotree.cpu().numpy()
    prmat2c_to_midi_file(prmat2c_np, f"exp/{data_type}_prmat2c.mid")
    estx_to_midi_file(pnotree_np, f"exp/{data_type}_pnotree.mid")
    prmat_to_midi_file(prmat, f"exp/{data_type}_prmat.mid")
    if chord is not None:
        chord_np = chord.cpu().numpy()
        chd_to_midi_file(chord_np, f"exp/{data_type}_chord.mid")
        return (
            prmat2c.to(device),
            pnotree.to(device),
            chord.to(device),
            prmat.to(device),
        )
    else:
        return prmat2c.to(device), pnotree.to(device), None, prmat.to(device)


def choose_song_from_val_dl(data_type, use_track=[0, 1, 2]):
    split_fpath = join(TRAIN_SPLIT_DIR, "pop909.pickle")
    with open(split_fpath, "rb") as f:
        split = pickle.load(f)
    print(split[1])
    num = int(input("choose one from pop909:"))
    song_fn = split[1][num]
    print(song_fn)

    song = DataSampleNpz(song_fn, use_track)
    return *get_data_preprocessed(song, data_type), song_fn


def choose_song_from_val_dl_musicalion(data_type):
    split_fpath = join(TRAIN_SPLIT_DIR, "musicalion.pickle")
    with open(split_fpath, "rb") as f:
        split = pickle.load(f)
    print(split[1])
    num = int(input("choose one from musicalion:"))
    song_fn = split[1][num]
    print(song_fn)

    song = DataSampleNpz_Musicalion(song_fn)
    return *get_data_preprocessed(song, data_type), song_fn


def get_autoreg_data(data: torch.Tensor, split_dim=1):
    steps = data.shape[split_dim]
    half_1, half_2 = data.split(steps // 2, dim=split_dim)
    half_1 = half_1.roll(-1, dims=0)
    # prmat_to_midi_file(half_1, f"exp/half1_rolled.mid")
    # prmat_to_midi_file(half_2, "exp/half2.mid")
    mid = torch.cat((half_2, half_1), dim=split_dim)
    # prmat_to_midi_file(mid, "exp/mid.mid")
    return mid


def get_mask(orig, inpaint_type, bar_list=None):
    B = orig.shape[0]
    if inpaint_type == "remaining":
        # just mask the existing notes
        mask = orig.clone()
    elif inpaint_type == "below":
        # inpaint the below area (inpaint accompaniment for melody)
        orig_onset = orig[:, 0, :, :]  # (, 128, 128)
        step_size = orig_onset.shape[1]
        pitch_size = orig_onset.shape[2]
        orig_onset = orig_onset.reshape((B * step_size, pitch_size))  # (steps, pitches)
        min_pitch = orig_onset.argmax(dim=1)  # (steps)
        # the first lowest pitch value
        first_min_pitch_idx = min_pitch.nonzero()[0]
        first_min_pitch = min_pitch[first_min_pitch_idx]
        for _ in range(first_min_pitch_idx):
            min_pitch[_] = first_min_pitch
        for idx in range(B * step_size):
            if min_pitch[idx] == 0:
                min_pitch[idx] = min_pitch[idx - 1]
        mask = torch.zeros_like(orig_onset)
        for step in range(B * step_size):
            mask[step, min_pitch[step] :] = 1
        mask = mask.reshape((B, step_size, pitch_size))

        mask = mask.unsqueeze(1)
        mask = mask.expand((-1, 2, -1, -1))  # (, 2, 128, 128)

    elif inpaint_type == "above":
        # inpaint the above area (inpaint melody for accompaniment)
        orig_onset = orig[:, 0, :, :]  # (, 128, 128)
        step_size = orig_onset.shape[1]
        pitch_size = orig_onset.shape[2]
        orig_onset = orig_onset.reshape((B * step_size, pitch_size))  # (steps, pitches)
        max_pitch = 127 - orig_onset.flip(1).argmax(dim=1)  # (steps)
        first_max_pitch_idx = max_pitch.nonzero()[0]
        first_max_pitch = max_pitch[first_max_pitch_idx]
        for _ in range(first_max_pitch_idx):
            max_pitch[_] = first_max_pitch
        for idx in range(B * step_size):
            if max_pitch[idx] == 127:
                max_pitch[idx] = max_pitch[idx - 1]
        mask = torch.zeros_like(orig_onset)
        for step in range(B * step_size):
            mask[step, 0 : max_pitch[step] + 1] = 1
        mask = mask.reshape((B, step_size, pitch_size))

        mask = mask.unsqueeze(1)
        mask = mask.expand((-1, 2, -1, -1))  # (, 2, 128, 128)

    elif inpaint_type == "bars":
        if bar_list is None:
            bar_list = input(
                "which bars would you like to inpaint for each 8-bar? (separate with ,): "
            )
            bar_list = [int(x) for x in bar_list.split(",")]
        mask = torch.ones_like(orig)
        for bar in bar_list:
            mask[:, :, bar * 16 : bar * 16 + 16, :] = 0
    else:
        raise NotImplementedError
    return mask

def rename_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("txt_enc", "text_enc")
        new_state_dict[new_key] = value
    return new_state_dict


class Experiments:
    def __init__(self, model_label, params, sampler: DiffusionSampler) -> None:
        self.model_label = model_label
        self.params = params
        self.sampler = sampler

    def predict(
        self,
        cond: torch.Tensor,
        cond_mid: Optional[torch.Tensor] = None,
        uncond_scale=1.0,
        autoreg=False,
        orig=None,
        mask=None,
        cond_concat=None,
    ):
        B = cond.shape[0]
        shape = [B, self.params.out_channels, self.params.img_h, self.params.img_w]
        # a bunch of -1
        uncond_cond = (-torch.ones([B, 1, self.params.d_cond])).to(device)
        print(f"generating {shape} with uncond_scale = {uncond_scale}")
        self.sampler.model.eval()
        if orig is None or mask is None:
            orig = torch.zeros(shape, device=device)
            mask = torch.zeros(shape, device=device)
        if args.ddim:
            t_idx = args.ddim_steps - 1
        else:
            t_idx = self.params.n_steps - 1
        noise = torch.randn(shape, device=device)
        with torch.no_grad():
            if autoreg:
                assert cond_mid is not None
                half_len = self.params.img_h // 2
                single_shape = [
                    1,
                    self.params.out_channels,
                    self.params.img_h,
                    self.params.img_w,
                ]
                orig_mid = get_autoreg_data(orig, split_dim=2)
                mask_mid = get_autoreg_data(mask, split_dim=2)
                noise_mid = get_autoreg_data(noise, split_dim=2)

                print(cond.shape)
                uncond_cond_seg = uncond_cond[0].unsqueeze(0)

                gen = []  # the generated
                for idx in range(B * 2 - 1):  # inpaint a 4-bar each time
                    if idx % 2 == 1:
                        cond_seg = cond_mid[idx // 2].unsqueeze(0)
                        orig_seg = orig_mid[idx // 2].unsqueeze(0)
                        mask_seg = mask_mid[idx // 2].unsqueeze(0)
                        noise_seg = noise_mid[idx // 2].unsqueeze(0)
                    else:
                        cond_seg = cond[idx // 2].unsqueeze(0)
                        orig_seg = orig[idx // 2].unsqueeze(0)
                        mask_seg = mask[idx // 2].unsqueeze(0)
                        noise_seg = noise[idx // 2].unsqueeze(0)
                    if idx != 0:
                        orig_seg[:, :, 0:half_len, :] = new_inpainted_half
                        mask_seg[:, :, 0:half_len, :] = 1
                    xt = self.sampler.q_sample(orig_seg, t_idx, noise_seg)
                    x0 = self.sampler.paint(
                        xt,
                        cond_seg,
                        t_idx,
                        orig=orig_seg,
                        mask=mask_seg,
                        orig_noise=noise_seg,
                        uncond_scale=uncond_scale,
                        uncond_cond=uncond_cond_seg,
                        cond_concat=cond_concat,
                        repaint_n=int(args.repaint_n),
                    )
                    if idx == 0:
                        gen.append(x0[:, :, 0:half_len, :])
                    # show_image(x0, f"exp/img/autoreg_{idx}.png")
                    # show_image(mask_seg, f"exp/img/autoreg_mask_{idx}.png", mask=True)

                    new_inpainted_half = x0[:, :, half_len:, :]
                    gen.append(new_inpainted_half)

                gen = torch.cat(gen, dim=0)
                print(f"piano_roll: {gen.shape}")
                assert gen.shape[0] == B * 2
                # gen = gen.view(n_samples, gen.shape[1], half_len * 2, gen.shape[-1])
                # print(f"piano_roll: {gen.shape}")

            else:
                # gen = self.sampler.sample(
                #     shape, cond, uncond_scale=uncond_scale, uncond_cond=uncond_cond
                # )
                xt = self.sampler.q_sample(orig, t_idx, noise)
                gen = self.sampler.paint(
                    xt,
                    cond,
                    t_idx,
                    orig=orig,
                    mask=mask,
                    orig_noise=noise,
                    uncond_scale=uncond_scale,
                    uncond_cond=uncond_cond,
                    cond_concat=cond_concat,
                    repaint_n=int(args.repaint_n),
                )
        # show_image(gen, "exp/img/gen.png")
        return gen

    def generate(
        self,
        cond: torch.Tensor,
        cond_mid: Optional[torch.Tensor] = None,
        uncond_scale=1.0,
        autoreg=False,
        polydis_recon=False,
        polydis_chd=None,
        no_output=False,
        cond_concat=None,
        output_dir="exp",
    ):
        gen = self.predict(
            cond,
            cond_mid,
            uncond_scale,
            autoreg,
            cond_concat=cond_concat,
        )

        if not no_output:
            output_stamp = (
                f"{self.model_label}"
                "["
                f"scale={uncond_scale}"
                f"{',autoreg' if autoreg else ''}"
                f"{',ddim' + str(args.ddim_steps) + '_eta' + str(args.ddim_eta) + '_' + str(args.ddim_discretize) if args.ddim else ''}"
                "]"
                f"_{datetime.now().strftime('%y-%m-%d_%H%M%S')}"
            )
            prmat2c = gen.cpu().numpy()
            prmat2c_to_midi_file(
                prmat2c, os.path.join(output_dir, f"{output_stamp}.mid")
            )
            if polydis_recon:
                aftertouch = PolydisAftertouch()
                prmat = prmat2c_to_prmat(prmat2c)
                print(prmat.shape)
                prmat_to_midi_file(
                    prmat, os.path.join(output_dir, f"{output_stamp}.mid")
                )
                prmat = torch.from_numpy(prmat)
                chd = polydis_chd
                aftertouch.reconstruct(prmat, chd, f"exp/{output_stamp}_recon.mid")
        return gen

    def inpaint(
        self,
        orig: torch.Tensor,
        inpaint_type,
        cond: torch.Tensor,
        cond_mid: Optional[torch.Tensor] = None,
        autoreg=False,
        orig_noise: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.0,
        bar_list=None,
        no_output=False,
        cond_concat=None,
        output_dir="exp",
    ):
        # show_image(orig, "exp/img/orig.png")
        orig_noise = orig_noise or torch.randn(orig.shape, device=device)
        mask = get_mask(orig, inpaint_type, bar_list)

        # show_image(mask, "exp/img/mask.png", mask=True)
        mask = mask.to(device)
        gen = self.predict(
            cond, cond_mid, uncond_scale, autoreg, orig, mask, cond_concat=cond_concat
        )

        if not no_output:
            output_stamp = (
                f"{self.model_label}_inp{args.repaint_n}_{inpaint_type}"
                "["
                f"scale={uncond_scale}"
                f"{',autoreg' if autoreg else ''}"
                f"{',ddim' + str(args.ddim_steps) + '_eta' + str(args.ddim_eta) + '_' + str(args.ddim_discretize) if args.ddim else ''}"
                "]"
                f"_{datetime.now().strftime('%y-%m-%d_%H%M%S')}"
            )
            prmat2c = gen.cpu().numpy()
            mask = mask.cpu().numpy()
            prmat2c_to_midi_file(
                prmat2c, os.path.join(output_dir, f"{output_stamp}.mid"), inp_mask=mask
            )
        return gen

    def show_q_imgs(self, prmat2c):
        if int(args.length) > 0:
            prmat2c = prmat2c[: int(args.length)]
        show_image(prmat2c, "exp/img/q0.png")
        for step in self.sampler.time_steps:
            s1 = step + 1
            if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
                noised = self.sampler.q_sample(prmat2c, step)
                show_image(noised, f"exp/img/q{s1}.png")

def custom_round(x):
    if x > 0.95 and x < 1.05:
        return 1
    else:
        return 0


def eval(text_input,key_target,time_target,tempo_target,random_numbers):
    if args.seed is not None:
        SEED = int(args.seed)
        print(f"fixed SEED = {SEED}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    text_path = os.path.join(args.output_dir, "text.json")
    key_path = os.path.join(args.output_dir, "key.json")
    ts_path = os.path.join(args.output_dir, "ts.json")
    tempo_path = os.path.join(args.output_dir, "tempo.json")
    with open(text_path, 'w') as f:
        json.dump(text_input, f)

    with open(key_path, 'w') as f:
        json.dump(key_target, f)

    with open(ts_path, 'w') as f:
        json.dump(time_target, f)

    with open(tempo_path, 'w') as f:
        json.dump(tempo_target, f)


    # params ready
    if args.custom_params_path is None:
        model_path = Path(args.chkpt_path).parent.parent
        if os.path.exists(f"{model_path}/params.yaml"):
            params_path = f"{model_path}/params.yaml"
        elif os.path.exists(f"{model_path}/params.json"):
            params_path = f"{model_path}/params.json"
        else:
            raise FileNotFoundError(
                f"params.yaml or params.json not found in {model_path}, please specify custom_params_path then."
            )
    else:
        params_path = args.custom_params_path
    params_path = convert_json_to_yaml(params_path)
    params = OmegaConf.load(params_path)
    model_label = params.model_name
    print(f"model_label: {model_label}")

    # model ready
    autoencoder = None
    unet_model = UNetModel(
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        channels=params.channels,
        attention_levels=params.attention_levels,
        n_res_blocks=params.n_res_blocks,
        channel_multipliers=params.channel_multipliers,
        n_heads=params.n_heads,
        tf_layers=params.tf_layers,
        d_cond=params.d_cond,
    )

    ldm_model = LatentDiffusion(
        linear_start=params.linear_start,
        linear_end=params.linear_end,
        n_steps=params.n_steps,
        latent_scaling_factor=params.latent_scaling_factor,
        autoencoder=autoencoder,
        unet_model=unet_model,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for i in range(int(args.num_generate)):
        # inpaint input ready
        prmat2c_inp = None
        print(f"""Generating song {i} of {int(args.num_generate)}""")
        if args.inpaint_type is not None:
            # choose the song to be inpainted
            print("getting the song to be inpainted...")
            if args.inpaint_from_midi is not None:
                song_fn_inp = args.inpaint_from_midi
                data_inp = get_data_for_single_midi(
                    args.inpaint_from_midi, "exp/chords_extracted_inpaint.out"
                )
                data_sample_inp = DataSample(data_inp)
                prmat2c_inp, _, _, _ = get_data_preprocessed(data_sample_inp, "inpaint")
            elif args.inpaint_from_dataset == "musicalion":
                prmat2c_inp, _, _, _, song_fn_inp = choose_song_from_val_dl_musicalion(
                    "inpaint"
                )  # here chd is None
            elif args.inpaint_from_dataset == "pop909":
                use_track_inp = [
                    int(x) for x in args.inpaint_pop909_use_track.split(",")
                ]
                prmat2c_inp, _, _, _, song_fn_inp = choose_song_from_val_dl(
                    "inpaint", use_track_inp
                )
            else:
                raise NotImplementedError
            print(f"Inpainting midi file: {song_fn_inp}")

        # condition data ready

        print("text controlled generation...")
        if int(args.length) > 0:
            length = int(args.length)
        elif prmat2c_inp is not None:
            length = prmat2c_inp.shape[0]
        else:
            length = int(input("how many 8-bars would you like to generate?"))
        prmat2c, pnotree, prmat, text = dummy_cond_input(length, params)

        text = args.text_caption


        # for demonstrating diffusion process
        if args.split_inpaint:
            print("only split prmat2c according to the inpainting type")
            mask = get_mask(orig=prmat2c_inp, inpaint_type=args.inpaint_type)
            prmat2c_to_midi_file(
                prmat2c, f"{args.from_midi[:-4]}_split.mid", inp_mask=mask
            )
            exit(0)

        # for polydis comparison
        if args.polydis:
            assert chd is not None
            assert prmat is not None
            if chd.shape[0] != prmat.shape[0]:
                min_size = min(chd.shape[0], prmat.shape[0])
                chd = chd[:min_size]
                prmat = prmat[:min_size]
            aftertouch = PolydisAftertouch()
            polydis_prmat = prmat.view(-1, 32, 128)
            print(polydis_prmat.shape)
            prmat_to_midi_file(polydis_prmat, "exp/polydis_prmat.mid")
            polydis_chd = chd.view(-1, 8, 36)  # 2-bars
            aftertouch.reconstruct(
                polydis_prmat,
                polydis_chd,
                "exp/polydis_gen.mid",
                chd_sample=args.polydis_chd_resample,
            )

        pnotree_enc, pnotree_dec = None, None
        chord_enc, chord_dec = None, None
        txt_enc = None
        text_enc = None
        if params.cond_type == "pnotree":
            pnotree_enc, pnotree_dec = load_pretrained_pnotree_enc_dec(
                PT_PNOTREE_PATH, 20
            )
            pnotree_enc, pnotree_dec = pnotree_enc.to(device), pnotree_dec.to(device)
        if "chord" in params.cond_type:
            if params.use_enc:
                chord_enc, chord_dec = load_pretrained_chd_enc_dec(
                    PT_CHD_8BAR_PATH,
                    params.chd_input_dim,
                    params.chd_z_input_dim,
                    params.chd_hidden_dim,
                    params.chd_z_dim,
                    params.chd_n_step,
                )
                chord_enc, chord_dec = chord_enc.to(device), chord_dec.to(device)
        if "txt" in params.cond_type:
            if params.use_enc:
                txt_enc = load_pretrained_txt_enc(
                    PT_POLYDIS_PATH,
                    params.txt_emb_size,
                    params.txt_hidden_dim,
                    params.txt_z_dim,
                    params.txt_num_channel,
                ).to(device)
        if "text" in params.cond_type:
            if params.use_enc:
                text_enc = load_pretrained_text_enc(
                ).to(device)

        if os.path.exists(f"{args.chkpt_path}/chkpts/{args.chkpt_name}"):
            args.chkpt_path = f"{args.chkpt_path}/chkpts/{args.chkpt_name}"
        if args.chkpt_path[-3:] == ".pt":
            # legacy ".pt" ckeckpoint
            model = Polyffusion_SDF.load_trained(
                ldm_model,
                args.chkpt_path,
                params.cond_type,
                params.cond_mode,
                chord_enc,
                chord_dec,
                pnotree_enc,
                pnotree_dec,
                txt_enc,
                text_enc,
            ).to(device)
        elif args.chkpt_path[-5:] == ".ckpt":
            # new lightning checkpoint
            init_model = Polyffusion_SDF(
                ldm_model,
                params.cond_type,
                params.cond_mode,
                chord_enc,
                chord_dec,
                pnotree_enc,
                pnotree_dec,
                txt_enc,
                text_enc
            )
            learner = LightningLearner.load_from_checkpoint(
                args.chkpt_path, model=init_model, optimizer=None
            )
            #learner = Polyffusion_SDF.load_trained(ldm_model,args.chkpt_path,"text")
            model = learner.model.to(device)
        else:
            raise RuntimeError
        if args.ddim:
            sampler = DDIMSampler(
                model.ldm,
                int(args.ddim_steps),
                args.ddim_discretize,
                int(args.ddim_eta),
                is_show_image=args.show_image,
            )
        else:
            sampler = SDFSampler(
                model.ldm,
                is_show_image=args.show_image,
            )
        expmt = Experiments(model_label, params, sampler)
        if args.only_q_imgs:
            expmt.show_q_imgs(prmat2c)
            exit(0)

        # encoded conditions ready
        polydis_chd = None
        cond_mid = None  # for autoregressive inpainting
        if params.cond_type == "pnotree":
            assert pnotree is not None
            cond = model._encode_pnotree(pnotree)
            if args.autoreg:
                cond_mid = model._encode_pnotree(get_autoreg_data(pnotree))
            pnotree_recon = model._decode_pnotree(cond)
            estx_to_midi_file(pnotree_recon, "exp/pnotree_recon.mid")
        elif params.cond_type == "chord":
            # print(chd.shape)
            assert chd is not None
            cond = model._encode_chord(chd)
            if args.autoreg:
                cond_mid = model._encode_chord(get_autoreg_data(chd))
            # print(chd_enc.shape)
            polydis_chd = chd.view(-1, 8, 36)  # 2-bars
            # print(polydis_chd.shape)
        elif params.cond_type == "txt":
            assert prmat is not None
            cond = model._encode_txt(prmat)
            if args.autoreg:
                cond_mid = model._encode_txt(get_autoreg_data(prmat))
        elif params.cond_type == "chord+txt":
            assert chd is not None
            assert prmat is not None
            if chd.shape[0] != prmat.shape[0]:
                min_size = min(chd.shape[0], prmat.shape[0])
                chd = chd[:min_size]
                prmat = prmat[:min_size]
            zchd = model._encode_chord(chd)
            ztxt = model._encode_txt(prmat)
            # print(chd_enc.shape)
            polydis_chd = chd.view(-1, 8, 36)  # 2-bars
            cond = torch.cat([zchd, ztxt], dim=-1)
            if args.autoreg:
                zchd_mid = model._encode_chord(get_autoreg_data(chd))
                ztxt_mid = model._encode_txt(get_autoreg_data(prmat))
                cond_mid = torch.cat([zchd_mid, ztxt_mid], dim=-1)
        elif params.cond_type == "text":
            cond = []
            for one_text in text_input:
                one_cond = model._encode_text(one_text)
                one_cond = torch.transpose(one_cond, 1, 2)
                cond.append(one_cond)

        else:
            raise NotImplementedError

        # concat conditioning
        cond_concat = None
        if hasattr(params, "concat_blurry") and params.concat_blurry:
            assert prmat2c is not None
            show_image(prmat2c, "exp/img/cond_concat_orig.png")
            cond_concat = get_blurry_image(prmat2c, params.concat_ratio)
            show_image(cond_concat, "exp/img/cond_concat.png")

        if params.cond_mode == "uncond":
            print("The model is trained unconditionally, ignoring conditions...")
            cond = -torch.ones_like(cond).to(device)

        if int(args.length) > 0:
            for index, one_cond in enumerate(cond, start=0):
                cond[index] = cond[index][: int(args.length)]
                print(f"selected cond shape: {cond[index].shape}")

        print("inpaint_type:", args.inpaint_type)

        # generate!
        if args.inpaint_type is not None:
            assert isinstance(prmat2c_inp, torch.Tensor)
            # crop shape
            if cond.shape[0] > prmat2c_inp.shape[0]:
                cond = cond[: prmat2c_inp.shape[0]]
            elif cond.shape[0] < prmat2c_inp.shape[0]:
                prmat2c_inp = prmat2c_inp[: cond.shape[0]]


            # inpaint!
            expmt.inpaint(
                orig=prmat2c_inp,
                inpaint_type=args.inpaint_type,
                cond=cond,
                cond_mid=cond_mid,
                autoreg=args.autoreg,
                orig_noise=None,
                uncond_scale=float(args.uncond_scale),
                cond_concat=cond_concat,
            )
        else:
            midi_list = []
            index_reminder = 0
            for one_cond in cond:
                print(f"Generating {index_reminder}-th song")
                gen = expmt.predict(
                    one_cond,
                    cond_mid,
                    float(args.uncond_scale),
                    args.autoreg,
                    cond_concat=cond_concat,
                )

                prmat2c = gen.cpu().numpy()
                if "Tensor" in str(type(prmat2c)):
                    prmat2c = prmat2c.cpu().detach().numpy()
                print(f"prmat2c : {prmat2c.shape}")

                labels = None
                is_custom_round = False
                inp_mask = None
                midi = pm.PrettyMIDI()
                piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
                origin = pm.Instrument(program=piano_program)
                inpainted = pm.Instrument(program=piano_program)
                t = 0
                n_step = prmat2c.shape[2]
                t_bar = int(n_step / 8)
                for bar_ind, bars in enumerate(prmat2c):
                    onset = bars[0]
                    sustain = bars[1]
                    for step_ind, step in enumerate(onset):
                        for key, on in enumerate(step):
                            if is_custom_round:
                                on = int(custom_round(on))
                            else:
                                on = int(round(on))
                            if on > 0:
                                dur = 1
                                while step_ind + dur < n_step:
                                    if not (int(round(sustain[step_ind + dur, key])) > 0):
                                        break
                                    dur += 1
                                note = pm.Note(
                                    velocity=80,
                                    pitch=key,
                                    start=t + step_ind * 1 / 8,
                                    end=min(t + (step_ind + dur) * 1 / 8, t + t_bar),
                                )
                                if inp_mask is not None:
                                    if inp_mask[bar_ind, 0, step_ind, key] == 0.0:
                                        inpainted.notes.append(note)
                                    else:
                                        origin.notes.append(note)
                                else:
                                    origin.notes.append(note)
                    t += t_bar
                midi.instruments.append(origin)
                if inp_mask is not None:
                    midi.instruments.append(inpainted)
                if labels is not None:
                    midi.lyrics.clear()
                    t = 0
                    for label in labels:
                        midi.lyrics.append(pm.Lyric(label, t))
                        t += t_bar
                midi_list.append(midi)
                fpath = os.path.join(args.output_dir, f"{random_numbers[index_reminder]}.mid")
                midi.write(fpath)
                index_reminder = index_reminder + 1

            # counter_key = 0
            # counter_ts = 0
            # counter_tempo = 0
            # counter_midi = 0
            # for one_midi in midi_list:
            #     print(one_midi.key_signature_changes)
            #     key = one_midi.key_signature_changes[0].key_number
            #     time_signature = f"{one_midi.time_signature_changes[0].numerator}/{one_midi.time_signature_changes[0].denominator}"
            #     tempo = one_midi.get_tempo_changes()[0]
            #     if key == key_target[counter_midi]:
            #         counter_key = counter_key + 1
            #     if time_signature == time_target[counter_midi]:
            #         counter_ts = counter_ts + 1
            #     if tempo == tempo_target[counter_midi]:
            #         counter_tempo = counter_tempo + 1
            #     counter_midi = counter_midi + 1
            #     output_stamp = (
            #         f"text_inp"
            #         "["
            #         f"{',ddim' + str(args.ddim_steps) + '_eta' + str(args.ddim_eta) + '_' + str(args.ddim_discretize) if args.ddim else ''}"
            #         "]"
            #         f"_{datetime.now().strftime('%y-%m-%d_%H%M%S')}"
            #     )
            #
            #
            #
            #
            # acc_key = counter_key/len(midi_list)
            # acc_tempo = counter_tempo/len(midi_list)
            # acc_ts = counter_ts/len(midi_list)
            #
            # print(f"Acc_key is {acc_key}")
            # print(f"Acc_tempo is {acc_tempo}")
            # print(f"Acc_ts is {acc_ts}")

def scale_to_index(scale):
    major_scales = {
        'C major': 0, 'C# major': 1, 'Db major':1, 'D major': 2, 'D# major': 3, 'Eb major': 3,
        'E major': 4, 'F major': 5, 'F# major': 6,'Gb major': 6, 'G major': 7,
        'G# major': 8, 'Ab major': 8, 'A major': 9, 'A# major': 10, 'Bb major': 10, 'B major': 11
    }

    minor_scales = {
        'C minor': 3, 'C# minor': 4,'Db minor': 4, 'D minor': 5, 'D# minor': 6, 'Eb minor': 6,
        'E minor': 7, 'F minor': 8, 'F# minor': 9, 'Gb minor': 9, 'G minor': 10,
        'G# minor': 11, 'Ab minor': 11, 'A minor': 0, 'A# minor': 1, 'Bb minor': 1, 'B minor': 2
    }

    if scale in major_scales:
        return major_scales[scale]
    elif scale in minor_scales:
        return minor_scales[scale]
    else:
        raise ValueError("未知的音阶名称: " + scale)

def generate_unique_random_numbers(start, end, count):
    if count > (end - start + 1):
        raise ValueError("指定的数量大于范围内的数字总数")
    return random.sample(range(start, end + 1), count)



if __name__ == "__main__":
    parser.add_argument("--chkpt_path",
                        default='/home/kinnryuu/ダウンロード/ICASSP_2025_Model/result/sdf_text/24-08-02_103135/chkpts/last.ckpt',
                        help="the path of the checkpoint to be used")
    parser.add_argument(
        "--custom_params_path",
        help="the path of custom parameters, default load from 'params.yaml/json' in the parent folder of the checkpoint",
    )
    parser.add_argument(
        "--uncond_scale",
        default=1.0,
        help="unconditional scale for classifier-free guidance",
    )
    parser.add_argument("--seed", help="use a specific seed for inference")
    parser.add_argument(
        "--autoreg",
        action="store_true",
        help="autoregressively inpaint the music segments",
    )
    parser.add_argument(
        "--from_dataset",
        default="pop909",
        help="choose condition from a dataset {pop909(default), musicalion}",
    )
    parser.add_argument(
        "--from_midi", help="choose condition from a specific midi file"
    )
    parser.add_argument(
        "--from_midi2",
        help="(incase two conditions are required) choose condition from a specific midi file",
    )
    parser.add_argument(
        "--inpaint_from_midi",
        help="choose the midi file for inpainting. if unspecified, use a song from dataset",
    )
    parser.add_argument(
        "--inpaint_from_dataset",
        default="pop909",
        help="inpaint a song from a dataset {pop909(default), musicalion}",
    )
    parser.add_argument(
        "--inpaint_pop909_use_track",
        default="0,1,2",
        help="which tracks to use as original song for inpainting (0: melody, 1: bridge, 2: piano accompaniment)",
    )
    parser.add_argument(
        "--inpaint_type", help="inpaint a song, type: {remaining, below, above, bars}"
    )
    parser.add_argument(
        "--ddim",
        action="store_true",
        help="whether to use DDIM sampler",
    )
    parser.add_argument(
        "--ddim_discretize",
        default="uniform",
        help="whether to use uniform or quad discretization in DDIM {uniform(default), quad}",
    )
    parser.add_argument("--ddim_eta", default=0.0, help="ddim eta, default: 0.0")
    parser.add_argument(
        "--ddim_steps",
        default=50,
        help="number of ddim sampling steps, default: 50",
    )
    parser.add_argument("--repaint_n", default=1, help="n sampling steps in RePaint")
    parser.add_argument("--length", default=8, help="the generated length (in 8-bars)")
    # you usually don't need to use the following args
    parser.add_argument(
        "--show_image",
        action="store_true",
        help="whether to show the images of generated piano-roll",
    )
    parser.add_argument(
        "--polydis_recon",
        action="store_true",
        help="whether to use polydis to reconstruct the generated midi from diffusion model",
    )
    parser.add_argument(
        "--chkpt_name",
        default="weights_best.pt",
        help="which specific checkpoint to use (default: weights_best.pt)",
    )
    parser.add_argument(
        "--only_q_imgs",
        action="store_true",
        help="only show q_sample results (for testing)",
    )
    parser.add_argument(
        "--split_inpaint",
        action="store_true",
        help="only split inpainted result according to the inpaint type (for testing). (inpaint: original, condition: inpainted)",
    )
    parser.add_argument(
        "--polydis",
        action="store_true",
        help="use polydis to generate MIDI. For comparison.",
    )
    parser.add_argument(
        "--polydis_chd_resample",
        action="store_true",
        help="Whether to resample chord with polydis generation",
    )
    parser.add_argument(
        "--num_generate", default=1, help="the number of samples to generate"
    )
    parser.add_argument(
        "--output_dir", default="exp", help="directory to store generated midis"
    )
    parser.add_argument(
        "--text_caption", help="text caption to describe music"
    )

    args = parser.parse_args()
    random_numbers = generate_unique_random_numbers(0, 160000, test_number)
    text_input = [captionset[i] for i in random_numbers]
    key_target = [scale_to_index(key_set[i]) for i in random_numbers]
    time_target = [ts_set[i] for i in random_numbers]
    tempo_target = [tempo_set[i] for i in random_numbers]
    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, 'w') as f:
        json.dump(random_numbers, f)
    eval(text_input,key_target,time_target,tempo_target,random_numbers)
