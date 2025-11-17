echo "开始测试......"

python ./main.py \
    --config_path ./config/STRGCN/Forecasting/STRGCN_Activity.yaml \
    --model_save_dir_path ./model_states/STRGCN/Activity \
    --result_save_dir_path ./result/STRGCN/Activity \

echo "结束测试......"
