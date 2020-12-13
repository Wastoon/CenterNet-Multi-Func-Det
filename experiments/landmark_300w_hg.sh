cd src
# train
python main.py --task landmark --exp_id hg_1x --dataset 300W --arch hourglass --batch_size 24 --master_batch 4 --lr 2.5e-4 --load_model ../models/ctdet_coco_hg.pth --gpus 3 --num_epochs 50 --lr_step 40
# test
python test.py --task landmark --exp_id hg_1x --dataset 300W --arch hourglass --keep_res --resume
# flip test
python test.py multi_pose --exp_id hg_1x --dataset coco_hp --arch hourglass --keep_res --resume --flip_test
cd ..

python demo.py --task landmark --load_model /data/mry/code/CenterNet/exp/landmark/rot_aug/model_best.pth --gpus 2 --dataset 300W --debug=2 --demo /data/mry/DataSet/300w/images/val --keep_res
