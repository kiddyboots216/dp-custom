from opacus.accountants.analysis import  rdp as privacy_analysis
import pickle
import os
from utils import parse_args

DEFAULT_ALPHAS = [1.0+x/100.0 for x in range(1,100)]+[2.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

def pre_compute_rdp(noise_multiplier):
    alphas = DEFAULT_ALPHAS
    file_name=f"rdp_calculation_{noise_multiplier:.3f}.dic"

    if os.path.exists(file_name):
        with open(file_name, 'rb') as handle:
            rdp_dict=pickle.load(handle)
    else:
        noise_multiplier=noise_multiplier
        rdp_dict=dict()
        print("pre-calculating rdp")
        PRECISION=2
        N_RATES=10**PRECISION
        for i in range(N_RATES+1):
            key_i = f"{0.0 + i/N_RATES:.{PRECISION}f}"
            rdp_dict[key_i]=privacy_analysis.compute_rdp(
                            q=0.0+i/N_RATES,
                            noise_multiplier=noise_multiplier,
                            steps=1,
                            orders=alphas,
                        )
            if i % int(N_RATES/10) == 0:
                print(f"RDP DICT[{key_i}] : {rdp_dict[key_i]}")
        rdp_dict['max']=rdp_dict[f"{1.000:.{PRECISION}f}"] # hardcoded to highest sampling rate
        with open(file_name, 'wb') as handle:
            pickle.dump(rdp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return rdp_dict

if __name__ == "__main__":
    args = parse_args()
    pre_compute_rdp(args.sigma)