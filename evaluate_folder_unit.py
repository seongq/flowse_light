import subprocess
import os
import re
# 체크포인트 파일들이 있는 폴더 경로


import random

gpu_num = input("gpu number 0,1중")
time_steps = ["uniform","gerkmann" ]
while True:
    ckpt_folders=["/workspace/flowse_KD/logs/dataset_WSJ0-CHiME3_derev_mode_orifirstflow_orisecondflow_orafirstflow_orasecondflow_kdfirstCFM_kdsecondCFM_nogradkd_CTFSE_noisy_mean_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_ym6md5q9","/workspace/flowse_KD/logs/dataset_WSJ0-CHiME3_derev_mode_orifirstflow_orisecondflow_orafirstflow_orasecondflow_kdfirstCFM_kdsecondCFM_nogradkd_CTFSE_noisy_mean_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_9n6qy09h","/workspace/flowse_KD/logs/dataset_WSJ0-CHiME3_derev_mode_orifirstflow_orisecondflow_orafirstflow_orasecondflow_kdfirstCFM_kdsecondCFM_nogradkd_CTFSE_noisy_mean_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_utr6k80b", "/workspace/flowse_KD/logs/dataset_WSJ0-CHiME3_derev_mode_orifirstflow_orisecondflow_orafirstflow_orasecondflow_kdfirstCFM_kdsecondCFM_nogradkd_CTFSE_noisy_mean_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_4120z2j9", "/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_ori_ora_kd_noisy_mean_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_3aprlqip","/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_local_align_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_b1jlofnf","/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_local_align_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_jmkolwl6", "/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_local_align_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_dr4453dh","/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_local_align_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_hh3156cz", "/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_local_align_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_6v55b6zx", "/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_local_align_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_k6blvq4c","/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_4wub2m1y","/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_6dgiky4l","/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_cv6dcxph","/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_p3j12744","/workspace/flowse_KD/logs/dataset_VCTK_corpus_mode_pesq_ori_ora_kd_noisy_mean_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_0othjng8"]
    
    # ["/workspace/flowse_KD/flowse_KD/logs/dataset_VCTK_corpus_mode_ori_ora_kd_zero_mean_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_eanojvs6","/workspace/flowse_KD/flowse_KD/logs/dataset_VCTK_corpus_mode_ori_ora_kd_zero_mean_no_grad_sigma_min_0.0_sigma_max_0.5_T_rev_1.0_t_eps_0.03_ay5bpm9w"]
    for ckpt_folder in ckpt_folders:
        # 정규표현식으로 dataset 이름 추출
        match = re.search(r"dataset_(.*?)_mode", ckpt_folder)
        if match:
            dataset_name = match.group(1)
            print(f"Extracted dataset name: {dataset_name}")
        else:
            print("Dataset name not found in the path.")

        test_dir = f"/workspace/datasets/{dataset_name}"
        if "CTFSE" in ckpt_folder:
            int_lists = ["1 1", "1 2","1 3",  "1 4", "1 5",  "1 6", "1 7","1 8","1 9","1 10"] # int_list 값들
        else:
            int_lists = ["1", "2","3",  "4", "5",  "6", "7","8","9","10"] # int_list 값들
        random.shuffle(int_lists)
        # ckpt 폴더에서 모든 .ckpt 파일 찾기
        ckpt_files = sorted([f for f in os.listdir(ckpt_folder) if f.endswith(".ckpt")])
        random.shuffle(ckpt_files)
        # 실행할 명령어 생성 및 실행
        for int_list in int_lists:
        
            for ckpt_file in ckpt_files:
                ckpt_path = os.path.join(ckpt_folder, ckpt_file)

                num_add = random.randint(0, 0)
                
                if num_add == 0 :
                    num_add =""
                else:
                    num_add = " "+str(num_add)
                for time_step in time_steps:
                 
                    cmd = f"CUDA_VISIBLE_DEVICES={gpu_num} python evaluate_cascading.py --ckpt {ckpt_path} --test_dir {test_dir} --int_list {int_list+num_add} --time_steps {time_step}"
                    print(f"Executing: {cmd}")
                    
                    process = subprocess.run(cmd, shell=True)

                    if process.returncode != 0:
                        print(f"Command failed: {cmd}")
                        break  # 실패 시에만 반복 종료
