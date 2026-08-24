import argparse
import json
import pathlib
import os
import multiprocessing as mp

from natsort import natsorted
from rich import print
import torch
import pickle

from general_motion_retargeting import (
    evaluate_retargeted_motion,
    retarget_offline_frames,
)
from general_motion_retargeting.utils.smpl import load_smplx_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel
import gc
import psutil
import tracemalloc


def check_memory(threshold_gb=2.0):
    mem = psutil.virtual_memory()
    used_memory_gb = (mem.total - mem.available) / (1024 ** 3)
    available_memory_gb = mem.available / (1024 ** 3)
    if available_memory_gb < threshold_gb:
        print(f"[WARNING] Memory usage:{used_memory_gb:.2f} GB, available:{available_memory_gb:.2f} GB, exceeding the threshold of {threshold_gb} GB.")
        return True
    return False


HERE = pathlib.Path(__file__).parent


def process_file(
    smplx_file_path,
    tgt_file_path,
    tgt_robot,
    SMPLX_FOLDER,
    tgt_folder,
    task_profile,
    device,
    quality_gate,
    minimum_available_memory_gb,
    total_files,
    verbose=False,
):
    def log_memory(message):
        if verbose:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
            print(f"[MEMORY] {message}: {memory_usage:.2f} GB")
    
    # Start memory tracking if verbose
    if verbose:
        tracemalloc.start()
        
    # Initial checks (with optional logging)
    log_memory("Initial memory usage")
    
    if check_memory(minimum_available_memory_gb):
        print(f"[SKIP] Insufficient memory for {smplx_file_path}.")
        return

    try:
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(smplx_file_path, SMPLX_FOLDER)
        log_memory("After loading SMPL-X data")
    except Exception as e:
        print(f"Error loading {smplx_file_path}: {e}")
        return
    
  
    tgt_fps = 30
    try:
        smplx_frame_data_list, aligned_fps = get_smplx_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return
    del body_model, smplx_output
    gc.collect()
    
    # retarget
    result = retarget_offline_frames(
        smplx_frame_data_list,
        src_human="smplx",
        tgt_robot=tgt_robot,
        actual_human_height=actual_human_height,
        task_profile=task_profile,
    )
    retargeter = result.retargeter
    qpos_list = result.qpos

    quality_report = evaluate_retargeted_motion(
        qpos_list,
        smplx_frame_data_list,
        retargeter,
        aligned_fps,
    )
    if quality_gate != "off":
        quality_path = f"{tgt_file_path}.quality.json"
        os.makedirs(os.path.dirname(quality_path), exist_ok=True)
        with open(quality_path, "w") as quality_file:
            json.dump(quality_report, quality_file, indent=2)
        if quality_gate == "reject" and not quality_report["quality_gate"]["passed"]:
            print(f"Rejected by quality gate: {smplx_file_path}")
            gc.collect()
            return

    log_memory("After retargeting")
    
    kinematics_model = KinematicsModel(retargeter.xml_file, device=device)

    try:
        root_pos = qpos_list[:, :3]
    except Exception as e:
        print(f"Error processing {smplx_file_path}: {e}")
        return
    root_rot = qpos_list[:, 3:7].copy()
    root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
    dof_pos = qpos_list[:, 7:]
    num_frames = root_pos.shape[0]

    fk_root_pos = torch.zeros((num_frames, 3), device=device)
    fk_root_rot = torch.zeros((num_frames, 4), device=device)
    fk_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        fk_root_pos, fk_root_rot, torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
    )

    log_memory("After forward kinematics")

    body_names = kinematics_model.body_names
    
    HEIGHT_ADJUST = True
    if HEIGHT_ADJUST:
        # height adjust to ensure the lowerset part is on the ground
        body_pos, _ = kinematics_model.forward_kinematics(torch.from_numpy(root_pos).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(root_rot).to(device=device, dtype=torch.float), 
                                                        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)) # TxNx3
        ground_offset = 0.0
        lowerst_height = torch.min(body_pos[..., 2]).item()
        root_pos[:, 2] = root_pos[:, 2] - lowerst_height + ground_offset # make sure motion on the ground
        
    ROOT_ORIGIN_OFFSET = True
    if ROOT_ORIGIN_OFFSET:
        # offset using the first frame
        root_pos[:, :2] -= root_pos[0, :2]
        
        
    motion_data = {
        "fps": aligned_fps,
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": body_names,
        "retarget_metadata": {
            "task_profile": retargeter.task_profile,
            "use_velocity_limit": result.use_velocity_limit,
            "passes_per_frame": result.passes_per_frame,
            "initial_settle_passes": result.initial_settle_passes,
            "joint_smoothing_kernel": result.joint_smoothing_kernel.tolist(),
            "root_orientation_smoothing_kernel": (
                result.root_orientation_smoothing_kernel.tolist()
            ),
            "joint_smoothing_passes": result.joint_smoothing_passes,
            "minimum_smoothing_alpha": float(result.smoothing_alpha.min()),
            "quality_gate": quality_report["quality_gate"],
        },
    }


    os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
    with open(tgt_file_path, "wb") as f:
        pickle.dump(motion_data, f)
        
    # Progress print based on tgt_folder
    done = 0
    for root, _, files in os.walk(tgt_folder):
        done += len([f for f in files if f.endswith('.pkl')])
    print(f"Processed {done}/{total_files}: {tgt_file_path}")
    
    if verbose:
        # Get memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\nTop 10 memory-consuming lines:")
        for stat in top_stats[:10]:
            print(stat)
        
        tracemalloc.stop()
        
    # clean cache
    torch.cuda.empty_cache()
    gc.collect()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument("--src_folder", type=str,
                        required=True,
                        )
    parser.add_argument("--tgt_folder", type=str,
                        required=True,
                        )
    
    parser.add_argument("--override", default=False, action="store_true")
    parser.add_argument("--num_cpus", default=1, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--body_model_dir",
        type=pathlib.Path,
        default=HERE / ".." / "assets" / "body_models",
    )
    parser.add_argument(
        "--minimum_available_memory_gb",
        type=float,
        default=2.0,
        help="skip a clip instead of risking OOM when less memory is available",
    )
    parser.add_argument(
        "--task_profile",
        default=None,
        help="override the robot's named task profile",
    )
    parser.add_argument(
        "--quality_gate",
        choices=("off", "report", "reject"),
        default="reject",
        help="write quality reports and optionally reject failed motions",
    )
    args = parser.parse_args()
    
    # print the total number of cpus and gpus
    print(f"Total CPUs: {mp.cpu_count()}")
    print(f"Using {args.num_cpus} CPUs.")
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    SMPLX_FOLDER = args.body_model_dir
    hard_motions_folder = HERE / ".." / "assets" / "hard_motions"

    verbose = False

    hard_motions_paths = [hard_motions_folder / "0.txt", 
                          hard_motions_folder / "1.txt"]
    hard_motions = []
    for hard_motions_path in hard_motions_paths:
        with open(hard_motions_path, "r") as f:
            for line in f:
                if "Motion:" in line:
                    motion_path = line.split(":")[1].strip()
                else:
                    continue
                motion_path = motion_path.split(",")[0].strip().split(".")[0]
                hard_motions.append(motion_path)
                
                
    args_list = []
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in natsorted(filenames):
            if filename.endswith("_stagei.npz"):
                continue
            if filename.endswith((".pkl", ".npz")):
                smplx_file_path = os.path.join(dirpath, filename)
                tgt_file_path = smplx_file_path.replace(src_folder, tgt_folder).replace(".npz", ".pkl")
                if not os.path.exists(tgt_file_path) or args.override:
                    args_list.append(
                        (
                            smplx_file_path,
                            tgt_file_path,
                            args.robot,
                            SMPLX_FOLDER,
                            tgt_folder,
                            args.task_profile,
                            args.device,
                            args.quality_gate,
                            args.minimum_available_memory_gb,
                        )
                    )
    print("full args_list:", len(args_list))
    
    # remove hard and infeasible motions
    exclude_file_content = ["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]
    
    new_args_list = []
    for arguments in args_list:
        motion_name = arguments[0].split("/")[-1].split('.')[0]
        if motion_name in hard_motions:
            continue
        if any(content in motion_name for content in exclude_file_content):
            continue
        new_args_list.append(arguments)
    args_list = new_args_list
    
    
    print("new args_list:", len(args_list))
    
    total_files = len(args_list)
    print(f"Total number of files to process: {total_files}")
    work = [arguments + (total_files, verbose) for arguments in args_list]
    if args.num_cpus == 1:
        for arguments in work:
            process_file(*arguments)
    else:
        with mp.Pool(args.num_cpus) as pool:
            pool.starmap(process_file, work)

    print("Done. Saved to ", tgt_folder)


if __name__ == "__main__":
    main()
