cd ../src
#reg&&Hm in common set
python test.py --task landmark --load_model /data/mry/code/CenterNet/exp/landmark/no_flip_aug/model_last.pth --gpus=1 --dataset 300W --test_split common --reg_or_reghm False

#reg#Hm in challenge
python test.py --task landmark --load_model /data/mry/code/CenterNet/exp/landmark/no_flip_aug/model_last.pth --gpus=1 --dataset 300W --test_split challenge --reg_or_reghm False

#reg&&Hm in full set
python test.py --task landmark --load_model /data/mry/code/CenterNet/exp/landmark/no_flip_aug/model_last.pth --gpus=1 --dataset 300W --test_split full --reg_or_reghm False

#reg_only in common set
python test.py --task landmark --load_model /data/mry/code/CenterNet/exp/landmark/no_flip_aug/model_last.pth --gpus=1 --dataset 300W --test_split common --reg_or_reghm True

#reg_only in challenge
python test.py --task landmark --load_model /data/mry/code/CenterNet/exp/landmark/no_flip_aug/model_last.pth --gpus=1 --dataset 300W --test_split challenge --reg_or_reghm True

#reg_only in full set
python test.py --task landmark --load_model /data/mry/code/CenterNet/exp/landmark/no_flip_aug/model_last.pth --gpus=1 --dataset 300W --test_split full --reg_or_reghm True