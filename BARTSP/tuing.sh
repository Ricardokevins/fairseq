export MKL_THREADING_LAYER=GUN
TOTAL_NUM_UPDATES=5000  
WARMUP_UPDATES=300      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=4
BART_PATH=/root/SheShuaijie/workspace/fairseq/Mytrain/bart.base/model.pt

CUDA_VISIBLE_DEVICES=0,1 fairseq-train data/cnn-bin \
    --seed 44 \
    --num-workers 5 \
    --no-progress-bar \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task contraBARTSp \
    --source-lang source \
    --positive_lang positive \
    --negtive_lang negtive \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_base \
    --criterion contraBARTSp_loss \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --no-epoch-checkpoints \
    --no-last-checkpoints \
    --save-dir /home/gengx/DiaSum/SamSum/Append \
    --find-unused-parameters;