<h1 align="center">SignBot: Learning Human-to-Humanoid Sign Language Interaction</h1>

<p align="center">
<h3 align="center"><a href="https://qiaoguanren.github.io/SignBot-demo/">Website</a> | <a href="https://arxiv.org/abs/2505.24266">arXiv</a> | <a href="https://www.bilibili.com/video/BV1u6jRzaE3L/?spm_id_from=333.337.search-card.all.click&vd_source=17e9bb674e80a4985637648c6e8e2205">Video</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
<img src="./video/signbot-demo-final-1.mp4" width="80%"/>
</p>

## Installation ##
```bash
conda create -n humanoid python=3.8
conda activate humanoid
cd
pip3 install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 -f https://download.pytorch.org/whl/cu121/torch_stable.html
git clone git@github.com:qiaoguanren/Signbot.git
cd Signbot
cd isaacgym/python && pip install -e .
cd ~/Signbot/rsl_rl && pip install -e .
cd ~/Signbot/legged_gym && pip install -e .
pip install "numpy==1.23.5" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask dill gdown
```

## Prepare dataset
1. Download from [here](https://github.com/2000ZRL/SOKE?tab=readme-ov-file) and extract the sign language datasets and SMPLX models.

2. Generate `.yaml` file for the motions you want to use. 
```bash
python generate_yaml.py
```
If you want to add more motions, you can use this script to generate the `.yaml` file.

3. Preprocess the body joint data and unify the input data format
```bash
python data_process_body.py
```

4. Preprocess the hand joint data with [hand_retargeting_tool](https://github.com/dexsuite/dex-retargeting). Please note that the environment dependencies for hand_retargeting_tool should not be placed together with Signbot; they should be handled separately.

```bash
cd dex-retargeting/example
python data_process_hand.py
```
we retarget linker hand with vector retargeting.

5. Retarget motions
```bash
cd ASE/ase/poselib
mkdir pkl retarget_npy
python retarget_motion_h1_all.py
```
This will retarget all motions in `ASE/ase/poselib/data/npy` to `ASE/ase/poselib/data/retarget_npy`.

6. Gnerate keybody positions

This step will require running simulation to extract more precise key body positions. 
```bash
cd legged_gym/legged_gym/scripts
python train.py debug --task h1_view --motion_name motions_debug.yaml --debug
```
Train for 1 iteration and kill the program to have a dummy model to load. 
```bash
python play.py debug --task h1_view --motion_name motions_autogen_all.yaml
```
It is recommended to use `motions_autogen_all.yaml` at the first time, so that later if you have a subset it is not neccessary to regenerate keybody positions. This will generate keybody positions to `ASE/ase/poselib/data/retarget_npy`.
Set wandb asset: 

## Usage 
To train a new policy
```bash
python train.py xxx-xx-some_descriptions_of_run --device cuda:0 --entity WANDB_ENTITY
```
`xxx-xx` is usually an id like `000-01`. `motion_type` and `motion_name` are defined in `legged_gym/legged_gym/envs/h1/h1_mimic_config.py`. They can be also given as arguments. Can set default WANDB_ENTITY in `legged_gym/legged_gym/utils/helpers.py`.

To play a policy
```bash
python play.py xxx-xx
```
No need to write the full experimentt id. The parser will auto match runs with first 6 strings (xxx-xx). So better make sure you don't reuse xxx-xx. Delay is added after 8k iters. If you want to play after 8k, add `--delay`.

To play with example pretrained models
```bash
python play.py 060-40 --delay --motion_name motions_debug.yaml
```
Try to press `+` or `-` to see different motions. The motion name will be printed on terminal. `motions_debug.yaml` is a small subset of motions for debugging and contains some representative motions.

## Sign Language Translator, Generator and Responder
For Sign Language Translator, we choose [Uni-Sign](https://arxiv.org/abs/2501.15187).
For Sign Language Responder, we provide you with `api.py` script to call local LLM.
For Sign Language Generator, we train it with Chinese and American hybrid sign language dataset. The Code has been open-sourced in this [repo](https://github.com/2000ZRL/SOKE?tab=readme-ov-file).

### Arguments
- --exptid: string, can be `xxx-xx-WHATEVER`, `xxx-xx` is typically numbers only. `WHATEVER` is the description of the run. 
- --device: can be `cuda:0`, `cpu`, etc.
- --delay: whether add delay or not.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --seed: random seed.
- --no_wandb: no wandb logging.
- --entity: specify wandb entity
- --web: used for playing on headless machines. It will forward a port with vscode and you can visualize seemlessly in vscode with your idle gpu or cpu. [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) vscode extension required, otherwise you can view it in any browser.
- --motion_name: e.g. `07_04` or `motions_all.yaml`. If `motions_all.yaml` is used, `motion_type` should be `yaml`.
- --motion_type: `single` or `yaml`
- --fix_base: fix the base of the robot.

For more arguments, refer `legged_gym/utils/helpers.py`.

### Cross-embodiment generalization
Due to hardware constraints, we completed the Sim-to-Real deployment on the wheeled robot Dexforce W1. The simulation engine used by W1 is [DexSim](https://github.com/DexForce/EmbodiChain).

### Acknowledgement
We derive the code framework from [Exbody](https://github.com/chengxuxin/expressive-humanoid/tree/main).

### Citation
```
@article{Qiao2025SignBotLH,
  title={SignBot: Learning Human-to-Humanoid Sign Language Interaction},
  author={Guanren Qiao and Sixu Lin and Ronglai Zuo and Zhizheng Wu and Kui Jia and Guiliang Liu},
  journal={ArXiv},
  year={2025},
  volume={abs/2505.24266}
}
```