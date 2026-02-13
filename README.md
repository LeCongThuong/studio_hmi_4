

python3 extract_per_tick_frames_from_csv.py \
  --root-dir /mnt/hdd/thuonglc/vsl/studio_vsl/recordings/di/session_20260114_161614/ \
  --out /mnt/hdd/thuonglc/vsl/studio_vsl/recordings/interact_test_14012026/frames \
  --cams all

python demo_2.py     --image_folder /mnt/hdd/thuonglc/vsl/studio_vsl/recordings/ttmt_spider_5/output_frames/100/     --output_folder /mnt/hdd/thuonglc/vsl/sam-3d-body/test_output_3     --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt     --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt     --detector_name sam3     --segmentor_name sam3 --debug

python triangulate_mhr70_subset.py \
    --mhr_py mhr70.py \
    --caliscope_toml config.toml \
    --cams left front__dev2 right \
    --npy_dirs /data/cam1/npy /data/cam2/npy /data/cam3/npy \
    --img_dirs /data/cam1/img /data/cam2/img /data/cam3/img \
    --out_dir /data/out_npz \
    --debug --debug_dir /data/debug_vis
    
    
python triangulate_mhr70_subset_v2.py     --mhr_py mhr70.py     --caliscope_toml config.toml     --cams left front right     --npy_dir /mnt/hdd/thuonglc/vsl/sam-3d-body/test_output_4/npy/     --img_dir /home/hmi/Downloads/100202026_tu/freestyle_thuong/frame_slow/200/     --out_npz triangulated.npz     --debug --debug_dir debug
   
python opt_pose_from_refined3d.py \
    --npz triangulated.npz \
    --npy_dir /mnt/hdd/thuonglc/vsl/sam-3d-body/test_output_4/npy/ \
    --hf_repo facebook/sam-3d-body-dinov3 \
     --cams left front right \
     --with_scale \
    --iters 200 --lr 0.05 \
    --debug_dir debug_opt \
    --out_npy opt_out.npy