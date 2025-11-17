echo "开始测试......"

python ./main.py \
    --config_path ./config/STRGCN/Forecasting/STRGCN_USHCN.yaml \
    --model_save_dir_path ./model_states/STRGCN/USHCN \
    --result_save_dir_path ./result/STRGCN/USHCN \

echo "结束测试......"
