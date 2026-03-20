# python train_res_transformer.py --name rtrans_retraining_full --gpu_id 1 --dataset_name t2m --batch_size 64 --vq_name rvq_retraining_full --cond_drop_prob 0.2 --share_weight 
python train_res_transformer.py --is_continue --name tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw --gpu_id 0 --dataset_name t2m --batch_size 10 --max_epoch 490 --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2 --cond_drop_prob 0.2 --share_weight 
# Originale epoche -> 440
# Nuovo epoche -> 460 (+20)
# Nuovo epoche -> 490 (+30)
