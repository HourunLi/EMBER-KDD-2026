# 完整版
python source_only_nifty.py run-seeds \
  --source-domain syn-2 \
  --target-domain syn-1 \
  --seeds 1 2 3 4 5 \
  --hidden 16 \
  --proj-hidden 16 \
  --epochs 500 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --sim-coeff 0.6 \
  --drop-edge-rate-1 0.001 \
  --drop-edge-rate-2 0.001 \
  --drop-feature-rate-1 0.1 \
  --drop-feature-rate-2 0.1 \
  --std-ddof 0 \
  --device cuda \
  --log-every 100

# bailA
python source_only_nifty.py run-seeds \
  --source-domain bailA_2 \
  --target-domain bailA_1 \
  --device cuda

# germanA
python source_only_nifty.py run-seeds \
  --source-domain germanA_2 \
  --target-domain germanA_1 \
  --device cuda

# syn
python source_only_nifty.py run-seeds \
  --source-domain syn-2 \
  --target-domain syn-1 \
  --device cuda

# pokec
python source_only_nifty.py run-seeds \
  --source-domain pokec_z \
  --target-domain pokec_n \
  --device cuda



