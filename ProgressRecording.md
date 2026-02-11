# ACT Investigating

### ACT Training on Dataset Trained from DexArt Sim Policy

- It seems when loading a single trajectory for training, ACT randomly pick a timestep and only load the observation (image and robot proprioception data) at that timestep and load all the actions after the timesteps (including the current timestep's action). This is how they "chunk" the action. But to make action sequences complete (to match the length of a standard trajectory in their dataset), they pad the actions with **0** to make the length of the action sequence after that timestep 400. 
  - **Why they do not pad the action with the last action? Which should be more consistent.**

- Also ACT is not utilizing qvel for the policy observation. Does qvel in observation make the training better or worse?

- ACT's own dataset has a constant episode length, while dexart dataset does not. How should we do with that?
  - We have to choose a constant horizon length to choose to chunk and pad the action sequence.



```
python imitate_dexart.py --ckpt_dir ./model/dexart_toilet --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --task_name dexart_toilet --seed 0 --object_obs flow
```

```
python imitate_dapg.py --ckpt_dir ./model/dapg_relocate --policy_class ACT --kl_weight 10 --chunk_size 10 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --task_name dapg_relocate --seed 0 --object_obs flow
```

