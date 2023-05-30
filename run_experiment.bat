call .venv/Scripts/activate.bat

python train_lgb_contrastive.py
python train_xgb_contrastive.py

python train_lgb.py
python train_xgb.py

python train_ensemble.py
python weight_ensemble.py