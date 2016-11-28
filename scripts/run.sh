unzip ../data/10000_lines.zip -d ../data/
rm -rf ../data/result/
nohup spark-submit \
    --master local[2] \
    ../src/tfidf.py > log.txt 2>&1 &
