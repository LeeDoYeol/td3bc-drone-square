```bash
git lfs install
git clone https://github.com/LeeDoYeol/td3bc-drone-square.git
cd td3bc-drone-square
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python train.py --data data/merged1.5M.csv.gz --steps 1000000 --device cuda

python eval_offline.py --data data/merged1.5M.csv.gz --episode 200

KMP_DUPLICATE_LIB_OK=TRUE python sim_eval.py --run_dir runs --seed 500
```
