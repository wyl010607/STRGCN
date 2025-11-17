echo "开始测试......"

python ./main.py \
    --config_path ./config/STRGCN/Forecasting/STRGCN_Physionet2012.yaml \
    --model_save_dir_path ./model_states/STRGCN/Physionet2012 \
    --result_save_dir_path ./result/STRGCN/Physionet2012 \

echo "结束测试......"
