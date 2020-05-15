python3 main.py --eval \
        --model resnet --lr 2e-6 --epoch 8 \
        --data_folder /local_storage/datasets/moein/Thorax/extracted/images \
        --checkpoints_path /Midgard/home/sorkhei/Thorax/checkpoints  \
        --results_path /Midgard/home/sorkhei/Thorax/results


python3 main.py --eval \
        --model resnet --freezed --lr 2e-6 --epoch 20 \
        --data_folder /local_storage/datasets/moein/Thorax/extracted/images \
        --checkpoints_path /Midgard/home/sorkhei/Thorax/checkpoints  \
        --results_path /Midgard/home/sorkhei/Thorax/results

python3 main.py --eval \
	--model googlenet --lr 2e-6 --epoch 23 \
	--data_folder /data/extracted/images \
	--checkpoints_path /checkpoints \
	--results_path /resulsts

