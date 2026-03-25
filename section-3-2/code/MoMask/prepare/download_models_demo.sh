cd /data
rm -rf checkpoints
mkdir checkpoints
cd checkpoints
mkdir t2m
cd t2m 
echo -e "Downloading pretrained models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing
unzip humanml3d_models.zip
rm humanml3d_models.zip
cd /home/user/app