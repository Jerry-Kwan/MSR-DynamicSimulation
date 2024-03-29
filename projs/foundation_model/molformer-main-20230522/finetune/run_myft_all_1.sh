python my_finetune_light.py \
        --device cuda \
        --batch_size 128  \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 3e-5 \
        --num_workers 8\
        --max_epochs 500\
        --num_feats 32 \
        --checkpoint_every 100 \
        --data_root ../data/qm9 \
        --seed_path ../data/checkpoints/N-Step-Checkpoint_3_30000.ckpt \
        --dataset_name qm9 \
        --measure_name u0 \
        --checkpoints_folder ./checkpoints_u0_all_1 \
        --all-params-ft \
        --torchmd-model transformer
