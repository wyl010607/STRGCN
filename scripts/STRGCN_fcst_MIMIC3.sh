echo "开始测试......"

python ./main.py \
    --config_path ./config/STRGCN/Forecasting/STRGCN_MIMIC3.yaml \
    --model_save_dir_path ./model_states/STRGCN/MIMIC3 \
    --result_save_dir_path ./result/STRGCN/MIMIC3 \

echo "结束测试......"
