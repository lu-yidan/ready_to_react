This repo is the official implementation of "Ready-to-React: Online Reaction Policy for Two-Character Interaction Generation".

<!-- [arxiv]() | [project page](https://zju3dv.github.io/ready_to_react/) -->


## 📰 News
[2025/02/20] We release the DuoBox dataset, training and evaluation code.

## 🛠️ Installation

<details>
<summary> Conda environment </summary>
```bash
conda create -n react python=3.9
conda activate react
# install pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# install pytroch3d
pip install pytorch3d-0.7.2-cp39-cp39-linux_x86_64.whl
# install other requirements
cat requirements.txt | sed -e '/^\s*-.*$/d' -e '/^\s*#.*$/d' -e '/^\s*$/d' | awk '{split($0, a, "#"); if (length(a) > 1) print a[1]; else print $0;}' | awk '{split($0, a, "@"); if (length(a) > 1) print a[2]; else print $0;}' | xargs -n 1 pip install
# install ready2react lib
pip install -e . --no-build-isolation --no-deps
```

NOTE:
`pytorch3d` [download link](https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1130/download.html).

</details>

## 💾 The DuoBox dataset

<details>
<summary>Download the DuoBox dataset</summary>

To download DuoBox, please fill in this [form](https://docs.google.com/forms/d/e/1FAIpQLSe8-G1PxykPoCR0wjH4HZKXkWz3_5UTa9x1if3L7bGPUXhmNA/viewform?usp=sharing). We provide skeleton extracted from OptiTrack and SMPL-X fitting results. In this project, we use the skeleton extracted from OptiTrack. The skeleton definition can be found in [link](https://docs.optitrack.com/motive/data-export/data-export-bvh). 

</details>

<details>
<summary>Dataset format</summary>
The data format of files under fbx_export_joints is:
```python
{
    f'{sbj_name}_position': # (T, J, 3), global position of joints
    f'{sbj_name}_rotation': # (T, J, 3, 3), [:, 1] are the root global orientations, the [:, 1:] are the other joints' local rotations relative to their parents
}
```

The data format of files under fbx_fit_smpl is:
```python
{
    f'{sbj_name}':
    {
        'transl':  # (T, 3)
        'global_orient': # (T, 3)
        'body_pose': # (T, 63)
        'betas': # (T, 10)
    }
}
```
</details>


## 🔧 Preparation
<details>
<summary>Preprocess data</summary>

1. Link DuoBox to data/:
```bash
mkdir data
ln -s /path/to/DuoBox data/DuoBox
```
2. Dataset spliting and preparing for reactive motion model:
```bash
python reactmotion/scripts/main.py -t data_test -c configs/exps/vqvae_tokenizer.yaml dataloader_cfg.dataset_cfg.gen_motion_split=True
python reactmotion/scripts/main.py -t data_test -c configs/exps/vqvae_tokenizer.yaml dataloader_cfg.dataset_cfg.gen_preprocess=True
python reactmotion/scripts/main.py -t data_test -c configs/exps/reactive_model.yaml dataloader_cfg.dataset_cfg.gen_preprocess=True
```
Files will be saved in data/boxing/reactive.

3. Dataset spliting and preparing for sparse control model:
```bash
python reactmotion/scripts/main.py -t data_test -c configs/exps/sparse_control.yaml dataloader_cfg.dataset_cfg.gen_motion_split=True
python reactmotion/scripts/main.py -t data_test -c configs/exps/sparse_control.yaml dataloader_cfg.dataset_cfg.gen_preprocess=True
```
</details>

<details>
<summary>Download pretrained models</summary>

Weights are shared in [link](https://drive.google.com/file/d/11H7-JoMobmnWFfx2TkF1TOKOBrxC4PQl/view?usp=sharing). Download and unzip under `data/`.

```bash
/data
|__ DuoBox
|__ trained_model
```
</details>

## 🚀 Usage

<details>
<summary>1️⃣ Reactive motion generation</summary>

```bash
python reactmotion/scripts/main.py -t test -c configs/exps/reactive_model.yaml model_cfg.inference_func=reactive
```
</details>

<details>
<summary>2️⃣ Two-character motion generation</summary>

```bash
python reactmotion/scripts/main.py -t test -c configs/exps/reactive_model.yaml model_cfg.inference_func=twoagent
```
</details>

<details>
<summary>3️⃣ Long-term two-character motion generation</summary>

```bash
python reactmotion/scripts/main.py -t test -c configs/exps/reactive_model.yaml model_cfg.inference_func=twoagent model_cfg.use_gt_length=False model_cfg.num_new_frames=1800
```
</details>

<details>
<summary>4️⃣ Sparse control</summary>

```bash
python reactmotion/scripts/main.py -t test -c configs/exps/sparse_control.yaml model_cfg.inference_func=reactive
```
</details>

<details>
<summary>Useful flags</summary>

```bash
model_cfg.inference_func=reactive or twoagent # inference type
runner_cfg.evaluator_cfg.feature_compute=False # skip eval FID
runner_cfg.evaluator_cfg.apd_compute=True # eval APD
```
</details>

<details>
<summary>Visualize results</summary>

Set the vis_output flag: 
```bash
runner_cfg.evaluator_cfg.vis_output=N # visualize number of N sequences
```

We use wis3d lib to visualize the results.
The visual results will be saved in `data/wis3d`.
In termial, run:
```bash
wis3d --vis_dir out/wis3d --host ${HOST} --port ${PORT}
```
You can then visualize the results in `${HOST}:${PORT}`.

</details>



## 🧠 Training the model by yourself

To train the models, simply set `-t train` istead of `-t test`. Don't forget to change the `exp_name=yourexpname`.

<details>
<summary>1️⃣ Train the VQ-VAE motion tokenizer</summary>

```bash
python reactmotion/scripts/main.py -t train -c configs/exps/vqvae_tokenizer.yaml exp_name=yourexpname1
```
</details>

<details>
<summary>2️⃣ Train the reactive motion model</summary>

Set the config `motoken_cfg_file` to the config file generated in step # 1. It should be saved in `data/record/yourexpname1`.
```bash
python reactmotion/scripts/main.py -t train -c configs/exps/reactive_model.yaml exp_name=yourexpname2
```
</details>


## 🙏 Acknowledgements
- [EasyVolCap](https://github.com/zju3dv/EasyVolcap) for their nice code base.
- [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) for the VQ-VAE architecture.


## 📚 Citation
If you use our model or dataset, please cite our Ready-to-React paper. If you use easyvolcap codebase, please cite EasyVolcap as follows.
```
@inproceedings{cen2025_ready_to_react,
  title={Ready-to-React: Online Reaction Policy for Two-Character Interaction Generation},
  author={Cen, Zhi and Pi, Huaijin and Peng, Sida and Shuai, Qing and Shen, Yujun and Bao, Hujun and Zhou, Xiaowei and Hu, Ruizhen},
  booktitle={ICLR},
  year={2025}
}
@article{xu2023easyvolcap,
  title={EasyVolcap: Accelerating Neural Volumetric Video Research},
  author={Xu, Zhen and Xie, Tao and Peng, Sida and Lin, Haotong and Shuai, Qing and Yu, Zhiyuan and He, Guangzhao and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  year={2023}
}

```
## 本机复现环境（lyd）

使用项目独立的 Python 3.9 `.venv`，核心版本为 PyTorch 1.13.0+cu116、
TorchVision 0.14.0+cu116、TorchAudio 0.13.0+cu116、PyTorch3D 0.7.2。
`requirements-compat.txt` 固定旧代码所需的 YAPF/NumPy 兼容版本；
`requirements-runtime.lock` 记录本次安装的其他依赖版本。

```bash
# 首次安装（需要 uv）；不会修改其他项目的环境
./scripts/install_env.sh
source .venv/bin/activate
```

DuoBox 需向作者申请并遵守数据许可，仅限非商业研究且不得再分发。
获批后将作者提供的压缩包解压为 `data/DuoBox/`，将官方权重解压为
`data/trained_model/`。数据、权重、缓存与运行日志均不提交 Git。

```text
data/
├── DuoBox/
│   ├── fbx_export_joints/       # 117 个原始关节 npy 文件
│   └── fbx_fit_smpl/
├── trained_model/
│   ├── vqvae_tokenizer/latest.pt
│   ├── reactive_model/latest.pt
│   └── sparse_control/latest.pt
└── boxing/                     # 预处理生成
```

```bash
# 首次预处理完整数据；再次运行会重新生成缓存
./scripts/prepare_data.sh

# 单个完整验证样本；不训练、不上传 W&B
./scripts/smoke_reactive.sh

# 完整长度的 reactive 推理（耗时更长）
WANDB_MODE=disabled .venv/bin/python reactmotion/scripts/main.py \
  -t test -c configs/exps/reactive_model.yaml \
  model_cfg.inference_func=reactive
```

预处理脚本设置 `PYTHONHASHSEED=0`，稳定原代码通过 `set` 得到的角色顺序。
它生成 reactive 和 sparse-control 两套缓存及相应归一化统计。
Smoke 关闭 FID/APD，只验证模型加载、生成和可视化输出，不代表论文指标复现。

查看可视化：

```bash
.venv/bin/wis3d --vis_dir data/wis3d --host 127.0.0.1 --port 19090
```

2026-10-09 本机验证（RTX 4090）：CUDA 矩阵乘法及反向传播、PyTorch3D CUDA
KNN 及反向传播、完整 reactive/vr3joints 数据预处理、reactive 与 sparse-control
各一个完整验证样本均通过。未启动训练，也未执行完整数据集指标评测。
本地验证日志位于 `data/setup_logs/`，生成文件哈希记录在
`data/setup_logs/artifacts.json`。

### PyCharm 单步学习（本机配置）

从 Show Apps 打开 PyCharm，打开 `/home/d080/workspace/ready_to_react`。
运行配置选择 `Ready-to-React | Debug reactive (1 sample)`，使用 Debug（Shift+F9）。
解释器固定为本项目 `.venv/bin/python`，工作目录为仓库根目录。
环境变量：`PYTHONHASHSEED=0`、`OMP_NUM_THREADS=4`、`WANDB_MODE=disabled`、
`PYTHONUNBUFFERED=1`、`PYDEVD_USE_FRAME_EVAL=NO`。
配置仅运行一个完整验证样本，数据加载 `num_workers=0`，关闭 FID/APD。

预设断点（行号对应当前源码）：

| 位置 | 观察内容 |
| --- | --- |
| `general_model.py:76` | 第一次预测前；`ts=4`、`self.ts_step=4` |
| `mocon_model.py:174` | 输入已组装；查看 `batch.latents.shape`、`batch.oppo_pose.shape`、`batch.pose_series.shape`、`batch.root_info.shape` |
| `mocon_model.py:158` | 扩散采样完成，准备解码；查看 `future_latents.shape` |
| `mocon_model.py:180` | 第一段的第一帧准备写回 agent；查看 `next_motion.shape`、`i` |

F8 执行当前行，F7 进入函数，Shift+F8 跳出函数，F9 运行到下一个断点。
初学时在扩散采样调用处用 F8，避免进入多次去噪循环；在 `pred_next` 处用 F7。
先看张量的 `.shape` 和少量切片，避免展开整个数据集。
IDE 设置与断点保存在本机 `.idea/`，通过 `.git/info/exclude` 排除，不修改模型源码。

### 浏览原始 DuoBox 动作

```bash
.venv/bin/python scripts/browse_duobox.py --port 19091
```

打开 `http://127.0.0.1:19091/`，从下拉菜单选择原始双人序列。
支持暂停、时间轴、0.5/1/2 倍速、拖动旋转及滚轮缩放。
从原始 120 Hz 数据每 4 帧取一帧，以 30 FPS 回放；双方均为数据集动作。
播放器仅绑定本机地址，不覆盖原数据或模型输出。
本机统计：117 段双人录制，共 3807.7 秒；单段 4.825–63.492 秒。

### Local reactive browser

Run from this repository using the existing environment:

```bash
.venv/bin/python scripts/reactive_browser.py
```

Open http://127.0.0.1:19094/. Select a raw DuoBox recording, the fixed actor,
start time, duration and seed, then generate the other actor using the released
reactive-model and tokenizer checkpoints. Generation runs in a subprocess with
one active job at a time; it is generated-then-played, not real-time streaming.

The adapter follows the original dataset preprocessing: Z-up to Y-up, global
rotations via FK, 120-to-30 Hz downsampling, and the initial one-model-frame
offset needed for velocities. Segment lengths round down to a multiple of four
frames and truncate at the recording end. The reacting actor's first four frames
come from the original counterpart, followed by autoregressive model output.
The fixed actor remains exactly the selected raw trajectory after preprocessing.
The playback selector switches between AI and the original counterpart at the
same frame, always showing two skeletons: orange fixed opponent and blue response.
Clip timing and seed are under the expandable settings panel. This is not G1
retargeting or physics simulation. Full raw recordings may include training data,
so this browser is not a held-out benchmark.

Each job saves its request, log, result JSON and motion NPZ under
`data/reactive_browser/<job-id>/`. NPZ arrays are Y-up positions in meters
with shape [T,21,3]: generated, fixed, reference; source_frames records the
corresponding original 120 Hz indices. The page can reopen its last job after
refresh and download the NPZ. A CUDA-capable configured environment is required.

### G1 因果流式响应导出

`scripts/reactive_stream_worker.py --request <job>/request.json` 复用发布的 reactive checkpoint，每次只向预测器提供已公开对手历史，输出 4 帧响应块，并等待下游 `ack_NNNN`。坐标从模型 Y-up 转回 DuoBox/GMR Z-up，输出全局关节位置和全局旋转。与 GMR 的 `scripts/reactive_g1_pipeline.py` 配合使用，浏览器入口在 19093 的 reactive_stream 版本。当前上层使用自身生成历史，不使用真实/仿真 G1 执行状态反馈；计算链路运行于虚拟时间。

### 因果流式生成保存节点（2026-10-14）

`scripts/reactive_stream_worker.py` 保留文件分块接口，并提供本地 Unix socket 接口供实时仿真使用。模型预热后常驻，每次生成 4 帧，只读取已释放的对手历史；支持固定种子的 normal/freeze/delay 输入对照。冻结/延迟同步变换位置与旋转后重新计算派生特征。实时网页入口位于相邻 GMR 仓库 `/live`，调度与物理跟踪位于部署仓库 `tools/reactive_realtime.py`。

300ms 播放缓冲由外部调度器选择，不属于模型固有配置。本次保存同时收录此前未提交的环境安装、数据预处理及响应浏览器脚本；不提交数据、模型、生成缓存和 Python 字节码。

### 功能总入口

收藏 http://127.0.0.1:19093/hub 。按功能分为 01 原始数据（19091）、02 人体响应（19094）、03 G1 结果回放（19093）、04 实时链路（19093/live）。四个页面顶部提供一致的同标签页导航并标记当前功能；旧地址与参数保留。回放版本的含义在总入口说明，reactive_rt3 的已完成实验回放与 /live 的边生成边仿真明确区分。各入口依赖对应本地服务，导航不启动或停止实验。

2026-10-14 验证：五个页面 HTTP 200，统一导航及当前页面标记检查通过；浏览器依次访问四个功能并返回总入口。仅修改导航与静态页面，未修改生成、重定向或控制链路。
