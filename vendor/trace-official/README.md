# TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning

This repository contains the official code for the paper [TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning](https://arxiv.org/abs/2601.21135).

## Citation

If you find this work useful, please cite:

```bibtex
@misc{fan2026tracetrajectoryrecoverycontinuous,
      title={TRACE: Trajectory Recovery for Continuous Mechanism Evolution in Causal Representation Learning},
      author={Shicheng Fan and Kun Zhang and Lu Cheng},
      year={2026},
      eprint={2601.21135},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.21135},
}
```

## Acknowledgments

Part of codebase is adapted from tdrl(https://github.com/weirayao/tdrl),
licensed under the MIT License.


## Installation

```bash
conda create -n trace python=3.8
pip install -e .
```

## Usage


### Data Generation

```bash
python -m trace.tools.gen_dataset -d 5 -s 222
python -m trace.tools.gen_dataset -d 10 -s 222
python -m trace.tools.gen_trajectory -d 5 -k 3
```
`-d` means the number of domains
`-k` means the number of active domains
`-s` means random seed

### Training

```bash
# Train with 5 domains
python scripts/train_change_5domain.py -e change_5 -s 222

# Train with 10 domains (recommend)
python scripts/train_change_10domain.py -e change_10 -s 222
```


### Inference
You need to change the config ./trace_crl/configs/inference.yaml first.
And then select the d k t:
```bash
python inference/infer.py -d 5 -k 3 -t simple
python inference/ablation_K_trajectory.py
python inference/ablation_W_recovery.py
python inference/eval_varying_K.py
```

## Dataset
UAVDT: https://www.kaggle.com/datasets/foryolotrain1/uavdt-2024
CMU mocap: https://mocap.cs.cmu.edu/

## License

MIT License
