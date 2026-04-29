# Dataset
We first collect ground-truth planning annotations from `LoTa-Bench`, and then follow the `E.T.` pipeline to render trajectories and create LMDB files for training and evaluation.

## LoTa-ALFRED


## Dataset in E.T. Format 
$ python -m alfred.gen.render_trajs
$ python -m alfred.data.create_lmdb