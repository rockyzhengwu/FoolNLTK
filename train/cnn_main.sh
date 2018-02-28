#!/usr/bin/env bash



# train file
TRAIN_FILE=./datasets/demo/train.txt
# dev file
DEV_FILE=./datasets/demo/dev.txt
# test file
TEST_FILE=./datasets/demo/test.txt

# data out dir
DATA_OUT_DIR=./datasets/demo

TAG_INDEX=1

# max length of sentlen
MAX_LENGTH=100

# 模型输出路径
MODEL_OUT_DIR=./results/demo_seg_cnn/


# word2vec
WORD2VEC=./third_party/word2vec/word2vec

###########################################################################
#
###########################################################################


VEC_OUT_DIR="$DATA_OUT_DIR"/vec/

CHAR_PRE_TRAIN_FILE="$VEC_OUT_DIR"/char_vec_train.txt
EMBEDING_FILE="$VEC_OUT_DIR"/char_vec.txt


# ckpt
CHECKPOINT_DIR="$MODEL_OUT_DIR"/ckpt/

# map  file
MAP_FILE="$DATA_OUT_DIR"/maps.pkl
# train, file
TRAIN_FILE_TF="$DATA_OUT_DIR"/train.tfrecord
# dev file
DEV_FILE_TF="$DATA_OUT_DIR"/dev.tfrecord
# test tf record
TEST_FILE_TF="$DATA_OUT_DIR"/test.tfrecord
# 保存部分大小参数
SIZE_FILE="$DATA_OUT_DIR"/size.json

if [ ! -d $DATA_OUT_DIR ];
    then mkdir -p $DATA_OUT_DIR;
fi;


if [ ! -d $VEC_OUT_DIR ];
  then mkdir -p $VEC_OUT_DIR
fi;


if [ "$1" = "vec" ]; then
  echo "train vec "
  python prepare_vec.py --train_file "$TRAIN_FILE" --dev_file "$DEV_FILE" --test_file "$TEST_FILE" --out_file "$CHAR_PRE_TRAIN_FILE"
  time $WORD2VEC -train "$CHAR_PRE_TRAIN_FILE" -output "$EMBEDING_FILE" -cbow 1 -size 100 -window 8 -negative 25 -hs 0 \
  -sample 1e-4 -threads 4 -binary 0 -iter 15 -min-count 5

elif [ "$1" = "map" ]; then

   echo "create map file"
   python create_map_file.py --train_file "$TRAIN_FILE" --embeding_file "$EMBEDING_FILE" --map_file "$MAP_FILE" \
   --size_file "$SIZE_FILE" --tag_index "$TAG_INDEX"

elif [ "$1" = "data" ]; then
   echo "data to tfrecord";
   python text_to_tfrecords.py --train_file "$TRAIN_FILE" --dev_file "$DEV_FILE" --test_file "$TEST_FILE" --map_file \
   "$MAP_FILE" --size_file "$SIZE_FILE" --out_dir "$DATA_OUT_DIR" --tag_index "$TAG_INDEX" --max_length "$MAX_LENGTH"

elif [ "$1" = "train" ]; then
    python cnn_train.py --train_file "$TRAIN_FILE_TF" --dev_file "$DEV_FILE_TF" --test_file "$TEST_FILE_TF"\
       --out_dir "$CHECKPOINT_DIR" --map_file "$MAP_FILE" --pre_embedding_file "$EMBEDING_FILE" --size_file "$SIZE_FILE"

elif [ "$1" = "export" ]; then
   echo "echo model"
   python export_model_cnn.py --checkpoint_dir "$CHECKPOINT_DIR" --out_dir "$MODEL_OUT_DIR"

else
   echo "param error"
fi
