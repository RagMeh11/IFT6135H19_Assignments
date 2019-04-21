#!/bin/sh

# Run vae and wgan for DCGAN architecture and Assignment Problem-2 architecture
python main_vae.py --cuda 1 --exp_name 'VAE_DCGAN' --niter 35 
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_DCGAN' --niter 35 
python main_vae.py --cuda 1 --exp_name 'VAE_AR' --generator_type 'Assignment_recom' --niter 35 
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_AR' --generator_type 'Assignment_recom' --niter 35 

python main_vae.py --cuda 1 --exp_name 'VAE_DCGAN' --niter 35 --init_epoch 34 --sample
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_DCGAN' --niter 35 --init_epoch 34 --sample
python main_vae.py --cuda 1 --exp_name 'VAE_AR' --generator_type 'Assignment_recom' --niter 35 --init_epoch 34 --sample
python main_wgan_gp.py --cuda 1 --exp_name 'WGAN_GP_AR' --generator_type 'Assignment_recom' --niter 35 --init_epoch 34 --sample

python score_fid.py "../Experiments_1/VAE_DCGAN/samples"
python score_fid.py "../Experiments_1/WGAN_GP_DCGAN/samples"
python score_fid.py "../Experiments_1/VAE_AR/samples"
python score_fid.py "../Experiments_1/WGAN_GP_AR/samples"