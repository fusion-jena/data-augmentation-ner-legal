#!/bin/bash

echo "starting augmentation"

COUNT=0


DIR="datasets/cBT1.0"

if [ -d "$DIR" ]; then
   echo "$DIR found directory"
else
   echo "$DIR NOT found."
fi

while ! [[ -d "$DIR" ]]
do
   NOW=$(date +"%T")
   echo "Restarting from checkpoint, $COUNT th-time, $NOW"
   python3 aug/backtranslate.py datasets/___1.0
   COUNT=$(($COUNT + 1))
   NOW=$(date +"%T")
   echo "Breaking Exception - sleeping for 10 minutes, $NOW"
   sleep 10m
done

echo "finished augmentation"