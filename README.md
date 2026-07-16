```bash
git lfs install
git clone https://github.com/LeeDoYeol/td3bc-drone-square.git
cd td3bc-drone-square
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 학습 (grad_clip으로 critic 안정화 + 체크포인트 저장)
python train.py --data data/merged1.5M.csv.gz --steps 1000000 --save_every 50000 --grad_clip 1.0 --device cuda

# 체크포인트 중 best 선택 + 궤적 그림 (오프라인 RL은 마지막 체크포인트가 최고가 아님)
KMP_DUPLICATE_LIB_OK=TRUE python select_best.py --run_dir runs --shapes line triangle square circle star --seed 500

# (선택) 오프라인 지표 + 최종 모델 5도형 평가
python eval_offline.py --data data/merged1.5M.csv.gz --episode 200
KMP_DUPLICATE_LIB_OK=TRUE python sim_eval.py --run_dir runs --seed 500
```
