#!/bin/bash
cd /Users/adam/interviews/kws/KWS_pytorch/data
TARGET=30000
curl -s "https://www.openslr.org/resources/87/mobvoi_hotword_dataset.tgz" | tar xz -C . &
tarpid=$!
while kill -0 $tarpid 2>/dev/null; do
  c=$(find mobvoi_hotword_dataset -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$c" -ge "$TARGET" ]; then
    kill $tarpid 2>/dev/null
    pkill -f "resources/87/mobvoi_hotword_dataset.tgz" 2>/dev/null
    break
  fi
  sleep 3
done
sleep 1
echo "DONE extracted=$(find mobvoi_hotword_dataset -name '*.wav' 2>/dev/null | wc -l | tr -d ' ') size=$(du -sh mobvoi_hotword_dataset 2>/dev/null | cut -f1)"
