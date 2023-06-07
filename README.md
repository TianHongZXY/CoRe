# Solving Math Word Problems via Cooperative Reasoning induced Language Models (ACL 2023)
## Data preparation
put the dataset under `data/`
## Fine tune generator
Set the hyperparameters in `train.slurm` and execute `bash train.slurm`
## Fine tune verifiers
Set the hyperparameters in `train_verifier.slurm` and execute `bash train_verifier.slurm`
## MCTS
After fine-tuning, specify the model path in `mcts.slurm`, execute `bash mcts.slurm`. Note that the provided script will not produce reasonable outputs unless the generator and verifiers are properly fine-tuned.

## Citation
Please consider citing our paper and starring this repo if you find them helpful. Thank you!
```bibtex
@article{zhu2022core,
         title={Solving Math Word Problem via Cooperative Reasoning induced Language Models},
         author={Zhu, Xinyu and Wang, Junjie and Zhang, Lin and Zhang, Yuxiang and Gan, Ruyi and Zhang, Jiaxing and Yang, Yujiu},
         journal={arXiv preprint arXiv:2210.16257},
         year={2022}
}
```

Feel free to open an issue if you have any questions.