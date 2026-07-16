```bash
git lfs install
git clone https://github.com/LeeDoYeol/td3bc-drone-square.git
cd td3bc-drone-square
git lfs pull
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 학습: 모든 도형(삼각형·사각형·원) 데이터, 1M step
#   --alpha 0.15 : 모방(BC) 비중을 크게 (불안정한 critic의 RL 항 영향 최소화)
#   --grad_clip  : critic 폭발 억제
#   --save_every : 체크포인트 저장 → 이후 best 선택
python train.py --data data/merged1.5M.csv.gz --steps 1000000 --save_every 25000 --alpha 0.15 --grad_clip 1.0 --device cuda

# 체크포인트 중 best 선택 + 5도형 궤적 그림
#   (오프라인 RL은 마지막 체크포인트가 최고가 아님 → 굴려보고 최저 오차 선택)
KMP_DUPLICATE_LIB_OK=TRUE python select_best.py --run_dir runs --shapes line triangle square circle star --seed 500

# (선택) 최종 모델 5도형 평가 그림
KMP_DUPLICATE_LIB_OK=TRUE python sim_eval.py --run_dir runs --seed 500
```
