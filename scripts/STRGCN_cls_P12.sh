echo "开始测试......"

python ./main.py \
    --config_path ./config/STRGCN/Classification/STRGCN_P12.yaml \
    --model_save_dir_path ./model_states/STRGCN/P12CLS \
    --result_save_dir_path ./result/STRGCN/P12CLS \

echo "结束测试......"
