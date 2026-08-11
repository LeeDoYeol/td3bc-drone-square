```bash
git lfs install
git clone https://github.com/LeeDoYeol/td3bc-drone-square.git
cd td3bc-drone-square
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p data
curl -L -o data/merged1.5M_hard_v2.csv.gz \
  https://media.githubusercontent.com/media/subsubli/drone_simulation/main/gym_pybullet_drones/gym_pybullet_drones/examples/data_hard_v2/merged1.5M_hard_v2.csv.gz

python gen_diffusion.py --data data/merged1.5M_hard_v2.csv.gz --n 1500000 --steps 30000 \
  --save_model gen_diff_hv2.pt --out synth_diff_hv2.npz --device cuda

python gen_gan.py --data data/merged1.5M_hard_v2.csv.gz --n 1500000 --steps 6000 --hidden 384 \
  --save_model gen_gan_hv2.pt --out synth_gan_hv2.npz --device cuda

python gen_check.py --data data/merged1.5M_hard_v2.csv.gz \
  --synth synth_diff_hv2.npz synth_gan_hv2.npz --out gen_check_hv2

python run_experiments.py --data data/merged1.5M_hard_v2.csv.gz \
  --diff synth_diff_hv2.npz --gan synth_gan_hv2.npz \
  --steps 300000 --save_every 10000 --att_d_gain 1.0 --device cuda

python collect_outputs.py --name hard_v2
```
