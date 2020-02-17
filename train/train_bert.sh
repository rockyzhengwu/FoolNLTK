
python ./train_bert_ner.py --data_dir=data/bid_train_data \
  --bert_config_file=./pretrainmodel/bert_config.json \
  --init_checkpoint=./pretrainmodel/bert_model.ckpt \
  --vocab_file=vocab.txt \
  --output_dir=./output/all_bid_result_dir/ --do_predict --do_export

