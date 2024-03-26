import sys, os, fnmatch
import subprocess
from multiprocessing import Pool

sys.path.insert(0,'/workspaces/wiggleformer/deps')

video_folder="/workspaces/wiggleformer/data/PANDA/videos/CLIN"
pred_out_dir="/workspaces/wiggleformer/data/PANDA/outputs/CLIN/openpose/annotations"
vis_out_dir="/workspaces/wiggleformer/data/PANDA/outputs/CLIN/openpose/vis"

OVERWRITE = False

def process_video(args):
    video_file, gpu_id = args
    video_file = os.path.join(video_folder, video_file)


    command = f"python /workspaces/wiggleformer/deps/mmpose/demo/inferencer_demo.py {video_file} \
              --pose2d human --pred-out-dir {pred_out_dir} \
              --vis-out-dir {vis_out_dir} --black-background --device cuda:{gpu_id} --skeleton-style openpose"
   
   #command = f"python /workspaces/wiggleformer/deps/mmpose/demo/inferencer_demo.py {video_file} \
   #           --pose3d human3d --pred-out-dir {pred_out_dir} \
   #           --vis-out-dir {vis_out_dir} --black-background --device cuda:{gpu_id}"

    subprocess.run(command, shell=True)

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def find_videos(directory):
    video_files = []

    for root, dirs, files in os.walk(directory):
        # Filtering for both .mp4 and .mov files, then combining the results
        mp4_files = fnmatch.filter([file.lower() for file in files], '*.mp4')
        mov_files = fnmatch.filter([file.lower() for file in files], '*.mov')
        all_files = mp4_files + mov_files

        print(all_files)
        for file in all_files:
            video_path = os.path.join(root, file)
            
            fname_base = os.path.basename(file).rsplit('.')[0]
            fname = f'{fname_base}.json'
            
            if not os.path.exists(os.path.join(vis_out_dir, fname)) and not OVERWRITE:
                print(f'Processing: {file}')
                video_files.append(video_path)

            if os.path.exists(os.path.join(vis_out_dir, fname)) and OVERWRITE:
                print(f'Overwriting: {file}')
                video_files.append(video_path)
    return video_files

if __name__ == '__main__':
    
    if not os.path.exists(pred_out_dir):
        os.makedirs(pred_out_dir)
        print(f'Created: {pred_out_dir}')
    if not os.path.exists(vis_out_dir):
        os.makedirs(vis_out_dir)
        print(f'Created: {vis_out_dir}')


    video_files = find_videos(video_folder)
    print(f'Processing: {len(video_files)} Files...')
    input(
        "Press Enter to continue..."
    )
    # Assuming you want to process 4 videos at a time on each GPU
    num_videos_per_gpu = 2
    num_gpus = 2

    # Create pairs of (video_file, gpu_id)
    tasks = [(video, i % num_gpus) for i, video_chunk in enumerate(chunker(video_files, num_videos_per_gpu)) for video in video_chunk]

    with Pool(num_gpus * num_videos_per_gpu) as p:
        p.map(process_video, tasks)
