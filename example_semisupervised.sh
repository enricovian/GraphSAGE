python -m graphsage.semisupervised_train \
--model graphsage_mean \
--train_prefix ./example_data/toy-ppi \
--learning_rate 0.02 \
--model_size big \
--epochs 2 \
--dropout 0.01 \
--weight_decay 0.0 \
--max_degree 128 \
--samples_1 25 \
--samples_2 10 \
--samples_3 0 \
--dim_1 128 \
--dim_2 128 \
--neg_sample_size 20 \
--batch_size 512 \
--identity_dim 0 \
--sigmoid True \
--base_log_dir ./logs \
--validate_iter 500 \
--validate_batch_size 256 \
--gpu 1 \
--print_every 5 \
#--random_context \
#--max_total_steps 10**10
