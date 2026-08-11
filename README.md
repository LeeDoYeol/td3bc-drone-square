```bash
git lfs install
git clone https://github.com/LeeDoYeol/td3bc-drone-square.git
cd td3bc-drone-square
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python run_all.py --device cuda
```

```bash
python run_all.py --device cuda --skip_small
python run_all.py --device cuda --only c1_real15 c3_real05_dif10
python run_all.py --help
```
