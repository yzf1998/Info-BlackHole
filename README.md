# Information Blackhole: Exploring Backdoor Mechanism in 3D Point Cloud Reconstruction

Official implementation of the paper *"Information Blackhole: Exploring Backdoor Mechanism in 3D Point Cloud Reconstruction"* by Zhifei Yang, Xiuping Liu, Junkai Qiu, and Yuhao Bian.

## 1. Requirements

**Environment:**  
- PyTorch 1.12.1, Python 3.8.20, CUDA 11.3, torchvision 0.13.1

**Installation:**
```bash
conda create -n info_blackhole python=3.8.20 -y && conda activate info_blackhole
pip install -r requirements.txt
```

**Optional Environment Variables:**
- `DATA_ROOT`: Root directory for datasets (default: `./data`)
- `CKPT_ROOT`: Root directory for model checkpoints (default: `./checkpoints`)


## 2. Datasets

### 2.1 Download

1. **ShapeNetPart**: Download from [shapenet.cs.stanford.edu](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip)
2. **ModelNet40**: Download from [modelnet.cs.princeton.edu](https://modelnet.cs.princeton.edu)

### 2.2 Directory Structure

```
data/
├── shapenet/shapenet_part/
│   └── shapenetcore_partanno_segmentation_benchmark_v0/
│       ├── 02691156/ ... 04379243/     # 16 synset directories
│       ├── train_test_split/
│       └── synsetoffset2category.txt
└── modelnet/
    └── modelnet40_normal_resampled/
        ├── airplane/ ... xbox/          # 40 class directories
        ├── modelnet10_shape_names.txt
        ├── modelnet40_train.txt
        └── modelnet40_test.txt
```

**Attack Target Shapes:**  
Pre-defined target point clouds are provided in `assets/target_pc/`.


## 3. Training

**Clean Model Training:**
```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m scripts.train_ae \
  --config configs/modelnet10/baseline.json
```

**Backdoored Model Training:**
```bash
# PointBA-I trigger (sphere) on ModelNet10
CUDA_VISIBLE_DEVICES=<GPU> python -m scripts.train_backdoor \
  --config configs/modelnet10/bd_modelnet10_sphere.json

# IRBA trigger (WLT) on ModelNet10
CUDA_VISIBLE_DEVICES=<GPU> python -m scripts.train_backdoor \
  --config configs/modelnet10/bd_modelnet10_wlt.json

# IBA trigger on ModelNet10
CUDA_VISIBLE_DEVICES=<GPU> python -m scripts.train_backdoor \
  --config configs/modelnet10/bd_modelnet10_iba.json
```

## 5. Evaluation

Evaluate a trained FoldNet checkpoint on clean and poisoned test sets:

```bash
CUDA_VISIBLE_DEVICES=<GPU> python -m scripts.eval_backdoor \
  --exp_dir snapshot/<experiment_id> \
  --epoch 300
```

## Reference

If you use this code in your research, please cite:

```bibtex
@misc{yang2026infoblackhole,
  title   = {Information Blackhole: Exploring the Backdoor Mechanism in 3D Point
             Cloud Reconstruction},
  author  = {Zhifei Yang and Xiuping Liu and Junkai Qiu and Yuhao Bian},
  year    = {2026}
}
```

## Acknowledgements

- [FoldingNet](https://github.com/qinglew/FoldingNet)
- [Diffusion Point Cloud](https://github.com/luost26/diffusion-point-cloud)


## License

MIT — see [LICENSE](LICENSE).
