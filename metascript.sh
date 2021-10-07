qsub -P P_humanum -l GPU=1,GPUtype=V100,sps=1 -pe multicores_gpu 4 -q mc_gpu_long -N off_sat /sps/humanum/user/sdo/AugmentedSocialScientist/saturation/off/ct.sh &
qsub -P P_humanum -l GPU=1,GPUtype=V100,sps=1 -pe multicores_gpu 4 -q mc_gpu_long -N endoexo_sat /sps/humanum/user/sdo/AugmentedSocialScientist/saturation/endoexo/ct.sh &
