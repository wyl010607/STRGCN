echo "开始测试......"

python ./main.py \
    --config_path ./config/STRGCN/Classification/STRGCN_Physionet.yaml \
    --model_save_dir_path ./model_states/STRGCN/PhysionetCLS \
    --result_save_dir_path ./result/STRGCN/PhysionetCLS \

echo "结束测试......"
