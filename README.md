```bash
git lfs install
git clone https://github.com/LeeDoYeol/td3bc-drone-square.git
cd td3bc-drone-square
git lfs pull
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python train.py --data data/merged1.5M.csv.gz --steps 1000000 --save_every 25000 --alpha 0.15 --grad_clip 1.0 --device cuda

KMP_DUPLICATE_LIB_OK=TRUE python select_best.py --run_dir runs --shapes line triangle square circle star --seed 500

KMP_DUPLICATE_LIB_OK=TRUE python sim_eval.py --run_dir runs --seed 500
```
