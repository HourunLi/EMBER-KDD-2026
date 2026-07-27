# epoch默认1000，但有可能过拟合
# 一次跑一个seed
# python source_only_gcn.py run \
#   --source-domain bailA_2 \
#   --target-domain bailA_1 \
#   --source-data-dir ./dataset/bailA \
#   --target-data-dir ./dataset/bailA \
#   --checkpoint ./checkpoints/source_only_gcn_bailA_2_seed1.pt \
#   --output-json ./results/source_only_gcn_bailA_2_to_bailA_1_seed1.json \
#   --hidden 16 \
#   --epochs 1000 \
#   --lr 1e-3 \
#   --weight-decay 1e-5 \
#   --seed 1 \
#   --device cuda \
#   --log-every 100

# bailA一次跑五个seed
python source_only_gcn.py run-seeds \
  --seeds 1 2 3 4 5 \
  --source-domain bailA_2 \
  --target-domain bailA_1 \
  --source-data-dir ./dataset/bailA \
  --target-data-dir ./dataset/bailA \
  --checkpoint-dir ./checkpoints \
  --results-dir ./results \
  --summary-json ./results/source_only_gcn_bailA_2_to_bailA_1_seeds1-5_summary.json \
  --hidden 16 \
  --epochs 1000 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --device cuda \
  --log-every 100 \
  --std-ddof 0

# germanA
python source_only_gcn.py run-seeds \
  --seeds 1 2 3 4 5 \
  --source-domain germanA_2 \
  --target-domain germanA_1 \
  --source-data-dir ./dataset/germanA \
  --target-data-dir ./dataset/germanA \
  --checkpoint-dir ./checkpoints/germanA \
  --results-dir ./results/germanA \
  --summary-json ./results/germanA/source_only_gcn_germanA_2_to_germanA_1_seeds1-5_summary.json \
  --hidden 16 \
  --epochs 500 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --std-ddof 0 \
  --device cuda \
  --log-every 100

# syn
python source_only_gcn.py run-seeds \
  --seeds 1 2 3 4 5 \
  --source-domain syn-2 \
  --target-domain syn-1 \
  --source-data-dir ./dataset/syn \
  --target-data-dir ./dataset/syn \
  --checkpoint-dir ./checkpoints/syn \
  --results-dir ./results/syn \
  --summary-json ./results/syn/source_only_gcn_syn-2_to_syn-1_seeds1-5_summary.json \
  --hidden 16 \
  --epochs 500 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --std-ddof 0 \
  --device cuda \
  --log-every 100

# pokec单seed
python source_only_gcn.py run \
  --source-domain pokec_z \
  --target-domain pokec_n \
  --hidden 16 \
  --epochs 1000 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --seed 1 \
  --device cuda

# pokec
python source_only_gcn.py run-seeds \
  --seeds 1 2 3 4 5 \
  --source-domain pokec_z \
  --target-domain pokec_n \
  --source-data-dir ./dataset/pokec_z \
  --target-data-dir ./dataset/pokec_n \
  --checkpoint-dir ./checkpoints/pokec \
  --results-dir ./results/pokec \
  --summary-json ./results/pokec/source_only_gcn_pokec_z_to_pokec_n_seeds1-5_summary.json \
  --hidden 16 \
  --epochs 50 \
  --lr 1e-3 \
  --weight-decay 1e-5 \
  --std-ddof 0 \
  --device cuda \
  --log-every 100








