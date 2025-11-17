echo "开始测试......"

python ./main.py \
    --config_path ./config/STRGCN/Classification/STRGCN_P19.yaml \
    --model_save_dir_path ./model_states/STRGCN/P19CLS \
    --result_save_dir_path ./result/STRGCN/P19CLS \

echo "结束测试......"
