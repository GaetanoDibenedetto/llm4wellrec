# python train_t2m_transformer.py --name mtrans_retraining_full --gpu_id 1 --dataset_name t2m --batch_size 64 --vq_name rvq_retraining_full 
python train_t2m_transformer.py --is_continue --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns --gpu_id 0 --dataset_name t2m --batch_size 10 --max_epoch 514 --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2
# Originale: 464
# +20 epoche rispetto a 484
# +30 epoche rispetto a 514