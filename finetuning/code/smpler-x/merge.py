import os
import shutil
import argparse
import numpy as np


def merge_npz_files(source_dir, output_file):
    files = sorted(os.listdir(source_dir))
    
    global_orient_arr = []
    body_poses_arr = []
    left_hand_arr = []
    right_hand_arr = []
    betas_arr = []
    transl_arr = []

    for file in files:
        if not file.endswith('.npz'):
            continue
        with np.load(os.path.join(source_dir, file)) as data:
            global_orient = data["global_orient"].tolist()
            body_pose = data["body_pose"].tolist()
            left_hand = data["left_hand_pose"].tolist()
            right_hand = data["right_hand_pose"].tolist()
            betas = data["betas"].tolist()
            transl = data["transl"].tolist()

            global_orient_arr.append(global_orient)
            body_poses_arr.append(body_pose)
            left_hand_arr.append(left_hand)
            right_hand_arr.append(right_hand)
            betas_arr.append(betas)
            transl_arr.append(transl)

    global_orientation_np = np.asarray(global_orient_arr)
    body_pose_np = np.asarray(body_poses_arr)
    left_hand_np = np.asarray(left_hand_arr)
    right_hand_np = np.asarray(right_hand_arr)
    betas_np = np.asanyarray(betas_arr)
    transl_np = np.asarray(transl_arr)

    np.savez(
        output_file,
        global_orient=global_orientation_np,
        body_pose=body_pose_np,
        left_hand_pose=left_hand_np,
        right_hand_pose=right_hand_np,
        betas=betas_np,
        transl=transl_np,
    )
    print(f"Merged npz files to {output_file}")

def call_inference(args, vid_filename):
    vid_name = vid_filename[:-4]
    frame_path = os.path.join(args.temp_path, vid_name, 'orig_img')
    log_path = os.path.join(args.temp_path, vid_name, 'process.log')
    video_path = os.path.join(args.input_path, vid_filename)

    os.makedirs(frame_path, exist_ok=True)
    if os.system(f'ffmpeg -i "{video_path}" -f image2 ' 
                f'-vf fps={args.fps} "{frame_path}/%06d.jpg" > "{log_path}" 2>&1') != 0:
        print(f"  -> Error extracting frames for {vid_filename}. See {log_path}")
        return
    
    if not os.listdir(frame_path):
        print(f"No frames extracted for {vid_filename}. Skipping.")
        return

    start_count = int(sorted(os.listdir(frame_path))[0].split('.')[0])
    end_count = len(os.listdir(frame_path)) + start_count - 1
    cmd_smplerx_inference = f'cd smplerx/main && python inference.py ' \
        f'--num_gpus 1 --pretrained_model {args.ckpt} ' \
        f'--agora_benchmark agora_model ' \
        f'--img_path "{frame_path}" --start {start_count} --end {end_count} ' \
        f'--output_folder "{args.temp_path}/{vid_name}" '\
        f'--show_verts --show_bbox --save_mesh >> "{log_path}" 2>&1'
    
    if os.system(cmd_smplerx_inference) != 0:
        print(f"  -> Error running inference for {vid_filename}. See {log_path}")
        return

    temp_smplx_path = os.path.join(args.temp_path, vid_name, 'smplx')
    final_npz_path = os.path.join(args.output_path, f'{vid_name}.npz')
    
    if os.path.exists(temp_smplx_path):
        os.makedirs(args.output_path, exist_ok=True)
        merge_npz_files(temp_smplx_path, final_npz_path)
    else:
        print(f"Warning: No smplx output found for {vid_filename}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str,
                        help='absolute path to input video folder',
                        default='/smplerx_inference/vid_input')
    parser.add_argument('--output_path', type=str,
                        help='absolute path to output folder',
                        default='/smplerx_inference/vid_output')
    parser.add_argument('--temp_path', type=str,
                        help='absolute path to temporary output folder',
                        default='/smplerx_inference/temp_output')

    args = parser.parse_args()
    args.format = 'mp4'
    args.fps = 20
    args.ckpt = 'smpler_x_h32'

    if not os.path.exists(args.input_path):
        print(f"Input path {args.input_path} does not exist.")
        exit(1)

    video_files = [f for f in os.listdir(args.input_path) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No .mp4 files found in {args.input_path}")
    
    total_videos = len(video_files)
    for i, vid in enumerate(video_files):
        print(f"Processing [{i+1}/{total_videos}] {vid}...", flush=True)
        call_inference(args, vid)