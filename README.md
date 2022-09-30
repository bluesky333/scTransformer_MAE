# scTransformer

## Dataset

1. Zhengmix8eq

   `/home/js4435/scratch60/scTransformer/Zhengmix8eq.h5ad`

2. Genomewide perturb-seq 

   `/home/js4435/scratch60/scTransformer/K562_gwps_normalized_singlecell_01.h5ad`

   `/home/js4435/scratch60/scTransformer/K562_essential_normalized_singlecell_01.h5ad`

   `/home/js4435/scratch60/scTransformer/rpe1_normalized_singlecell_01.h5ad`

#### Format

#### .csv

1. Gene expression matrix: 

   A **gene by cell** matrix stored in .csv file, with row names (gene name) and column names (cell names)

2. Meta data:

   A **cell by feature** matrix stored in .csv file.

#### .h5ad

1. 

## Training

### Vanilla Training

```
python main_pretrain.py \
    --output_dir 0717_Zhengmix8eq \
    --log_dir 0717_Zhengmix8eq_tensorboard \
    --expr_path /content/drive/Shareddrives/Documentation/Jie_Sheng/Zhengmix8eq/zhengmix8eq_scaleddata.csv \
    --meta_path /content/drive/Shareddrives/Documentation/Jie_Sheng/Zhengmix8eq/meta.csv \
    --label_name x \
    --batch_size 8 \
    --model mae_vit_d64 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 2000 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --gene_embed_dim 64
```

### Multi-node Training

```
python submitit_pretrain.py \
    --nodes 1 \
    --ngpus 2 \
    --timeout 10080 \
    --batch_size 8 \
    --model mae_vit_d64 \
    --mask_ratio 0.75 \
    --epochs 500 \
    --warmup_epochs 100 \
    --blr 0.5e-6 \
    --weight_decay 0.05 \
    --file_type 'h5ad' \
    --h5ad_path '/home/js4435/scratch60/K562_gwps_normalized_singlecell_01.h5ad' \
    --label_name gene \
```

* Farnam submission

  ```
  sbatch submit_K562.slurm
  ```

  `submit_K562.slurm`

  ```
  #!/bin/bash
  
  #SBATCH --mail-user jie.sheng@yale.edu
  #SBATCH --job-name=submission
  #SBATCH --output=log.txt
  #SBATCH -p pi_dijk
  #SBATCH --mail-type=ALL
  #SBATCH --time=5:00
  #SBATCH --cpus-per-task=1
  
  master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  export MASTER_ADDR=$master_addr
  echo "MASTER_ADDR="$MASTER_ADDR
  
  python submitit_pretrain.py \
      --job_dir '' \
      --nodes 1 \
      --ngpus 2 \
      --timeout 10080 \
      --batch_size 4 \
      --model mae_vit_d64 \
      --norm_pix_loss \
      --mask_ratio 0.75 \
      --epochs 500 \
      --warmup_epochs 100 \
      --blr 0.5e-6 --weight_decay 0.05 \
      --file_type 'h5ad' \
      --h5ad_path '/home/js4435/scratch60/K562_gwps_normalized_singlecell_01.h5ad' \
      --label_name gene \
      --partition pi_dijk \
  ```
