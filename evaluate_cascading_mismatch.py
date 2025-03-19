import time
import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from torchaudio import load
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd
import sys
from flowmse.data_module import SpecsDataModule
from flowmse.model import VFModel
import pdb
import os
from flowmse.util.other import pad_spec
from flowmse.sampling import get_white_box_solver
from utils import energy_ratios, ensure_dir, print_mean_std

import pdb
torch.set_num_threads(5)
torch.cuda.empty_cache()
if __name__ == '__main__':
    parser = ArgumentParser()
   

    parser.add_argument("--odesolver", type=str,
                        default="euler", help="euler")
    parser.add_argument("--reverse_starting_point", type=float, default=1.0, help="Starting point for the reverse SDE.")
    parser.add_argument("--reverse_end_point", type=float, default=0.03)
    
    parser.add_argument("--test_dir")
    parser.add_argument("--ckpt", type=str, help='Path to model checkpoint.')
    parser.add_argument("--int_list", type=int, nargs='+', help="List of integers")


    args = parser.parse_args()

    
    dataset_name= os.path.basename(os.path.normpath(args.test_dir))
    if dataset_name=="WSJ0-CHiME3":
        mismatch_dataset =  "VCTK_corpus"
    elif dataset_name=="VCTK_corpus":
        mismatch_dataset =  "WSJ0-CHiME3"
    
    
    checkpoint_file = args.ckpt
    int_list = "_".join(map(str, args.int_list))
    # raise("target_dir 부터 확인해")
    

    # Settings
    sr = 16000
    print(args.int_list)
    odesolver = args.odesolver
    int_list = args.int_list
    
    
    # Load score model
    try:
        model = VFModel.load_from_checkpoint(
            checkpoint_file, base_dir="",
            batch_size=8, num_workers=4, kwargs=dict(gpu=False)
        )
    except:
        model = VFModel_SGMSE_CRP.load_from_checkpoint(
            checkpoint_file, base_dir="", batch_size=8, num_workers=4, kwargs=dict(gpu=False)
        )
    int_list_str = "_".join(map(str, args.int_list))
    import re

    def extract_epoch(checkpoint_file):
        match = re.search(r'epoch=([0-9]{1,4})', checkpoint_file)
        if match:
            return int(match.group(1))  # 정수로 변환하여 반환
        return None  # 매칭되지 않으면 None 반환

    # 예제 경로
    match = re.search(r"mode_([a-zA-Z0-9]+)", checkpoint_file)

    # 결과 출력
    if match:
        mode_value = match.group(1)
        print(mode_value)
    else:
        print("No match found")
    epoch_number = extract_epoch(checkpoint_file)
    target_dir = f"/workspace/results/clean_KD_tiny/Mismatch_{mismatch_dataset}_to_{dataset_name}_mode_{model.mode_condition}_epoch_{epoch_number}_{int_list_str}/"
    results_candidate_path = os.path.join(target_dir, "_avg_results.txt")
    if os.path.exists(results_candidate_path):  # 파일 존재 여부 확인
        print(f"파일이 존재하므로 프로그램을 종료합니다: {results_candidate_path}")
        sys.exit()  # 프로그램 종료
    print("evaluation sijak")
    ensure_dir(target_dir + "files/")
    reverse_starting_point = args.reverse_starting_point
    reverse_end_point = args.reverse_end_point
        
    model.ode.T_rev = reverse_starting_point
        
    
    model.eval(no_ema=False)
    model.cuda()
    clean_dir = join(args.test_dir, "test", "clean")
    noisy_dir = join(args.test_dir, "test", "noisy")
    noisy_files = sorted(glob.glob('{}/*.wav'.format(noisy_dir)))
    




    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
    for cnt, noisy_file in tqdm(enumerate(noisy_files)):
        filename = noisy_file.split('/')[-1]
        
        # Load wav
        x, _ = load(join(clean_dir, filename))
        y, _ = load(noisy_file)

        #pdb.set_trace()        

         
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor

        
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        Y = Y.cuda()
        with torch.no_grad():
            for i in range(len(int_list)):
                N = int_list[i]
                if N==0:
                    continue
                if i == 0:
                    xt, _ = model.ode.prior_sampling(Y.shape, Y)
                    CONDITION = Y
                    ENHANCED = Y
                else:
                    ENHANCED = xt
                    xt, _ = model.ode.prior_sampling(Y.shape,ENHANCED)
                   
                    CONDITION = ENHANCED
                   
                    
                xt = xt.to(Y.device)
                timesteps = torch.linspace(reverse_starting_point, reverse_end_point, N, device=Y.device)
                for i in range(len(timesteps)):
                    t = timesteps[i]
                    if i == len(timesteps)-1:
                        dt = 0-t
                    else:
                        dt = timesteps[i+1]-t
                    vect = torch.ones(Y.shape[0], device=Y.device)*t
                    xt = xt + dt * model(xt, vect, ENHANCED, CONDITION)            
                
        
        sample = xt.clone()
        
        
        sample = sample.squeeze()
        
        x_hat = model.to_audio(sample, T_orig)
        # print("완료")
        y = y * norm_factor
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        
      
        # Convert to numpy
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()
        n = y - x

        # Write enhanced wav file
        write(target_dir + "files/" + filename, x_hat, 16000)

        # Append metrics to data frame
        data["filename"].append(filename)
        try:
            p = pesq(sr, x, x_hat, 'wb')
        except: 
            p = float("nan")
        data["pesq"].append(p)
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])

    # Save results as DataFrame
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)

    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        
        file.write("odesolver: {}\n".format(odesolver))
       
        file.write("N: {}\n".format(N))
        
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))
        file.write("Reverse end point: {}\n".format(reverse_end_point))
        # file.write("Mismatch: Mismatch")
        file.write("trained_data: {}\n".format(mismatch_dataset))
        file.write("data: {}\n".format(args.test_dir))
        file.write("epoch: {}\n".format(epoch_number))
        file.write("evaluationnumbers: {}\n".format(int_list_str))
        file.write("mode: {}\n".format(model.mode_condition))
        