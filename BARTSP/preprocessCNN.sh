
TASK=/root/SheShuaijie/workspace/fairseq/BARTSP/data/cnn
for SPLIT in train val
do
  for LANG in source positive negtive
  do
    python3 -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done



fairseq-preprocess \
  --source-lang "source" \
  --target-lang "positive" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict /root/SheShuaijie/workspace/fairseq/Mytrain/bart.base/dict.txt \
  --tgtdict /root/SheShuaijie/workspace/fairseq/Mytrain/bart.base/dict.txt;

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "negtive" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict /root/SheShuaijie/workspace/fairseq/Mytrain/bart.base/dict.txt \
  --tgtdict /root/SheShuaijie/workspace/fairseq/Mytrain/bart.base/dict.txt;

