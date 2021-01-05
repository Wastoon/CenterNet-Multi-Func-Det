#!/bin/bash
python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_142311_894.avi \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth   \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --test_scales "0.5,0.6,0.7,0.8,1.0,1.1,1.2" \
                    --nms \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_142311_894.avi  ##52

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_143042_958.avi  \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_143042_958.avi \
                    --test_scales "0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2" \
                    --nms
python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_143336_221.avi  \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_143336_221.avi \
                    --test_scales "0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_143641_245.avi  \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_143641_245.avi \
                    --test_scales "0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_145426_090.avi  \
                    --load_model /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_angle_all_vehicle_wh_adjust_no_rotatebbox_batch3/model_last_good.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_145426_090.avi \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_175044_886.avi  \
                    --load_model /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_angle_all_vehicle_wh_adjust_no_rotatebbox_batch3/model_last_good.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_175044_886.avi \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_175349_992.avi  \
                    --load_model /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_angle_all_vehicle_wh_adjust_no_rotatebbox_batch3/model_last_good.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_175349_992.avi \
                    --test_scales "0.8,0.9,1.0,1.1,1.2,1.3,1.4, 1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_180820_519.avi  \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_180820_519.avi \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_181134_293.avi  \
                    --load_model /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth   \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_181134_293.avi \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_181431_269.avi  \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-08_181431_269.avi \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-09_095532_076.avi  \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-09_095532_076.avi \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-09_095830_741.avi  \
                    --load_model /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_3_1_2/model_20.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-09_095830_741.avi  \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-09_100653_126.avi  \
                    --load_model /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_angle_all_vehicle_wh_adjust_no_rotatebbox_batch3/model_last_good.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-09_100653_126.avi  \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-09_101232_238.avi  \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_3_1_2/model_5.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-09_101232_238.avi  \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-08_155544_431.avi  \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-09_095532_076.avi \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms

python src/demo.py --gpus 0\
                    --demo /data4/mry/vehicle_inference/input/2020-12-09_100158_835.avi  \
                    --load_model  /data1/mry/code/centernet_newversion/exp/vehicle_det/dla34_dcn_angle_all_vehicle_wh_adjust_no_rotatebbox_batch4_trainscale_0_1_1_2/model_5.pth  \
                    --task vehicle_det \
                    --dataset ALLVehicle \
                    --debug 5 \
                    --arch dla_34   \
                    --vis_thresh 0.3 \
                    --center_thresh 0.3 \
                    --show_label \
                    --output_video_demo /data4/mry/vehicle_inference/output/2020-12-09_095532_076.avi \
                    --test_scales "0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5" \
                    --nms