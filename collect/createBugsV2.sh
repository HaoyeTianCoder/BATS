dir=/Users/haoye.tian/Documents/University/project/defects4j_buggy/

proj=Chart
for bugId in $(seq 1 26)
do
	defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Cli
for bugId in $(seq 1 40)
do
	defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Closure
for bugId in $(seq 1 176)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Codec
for bugId in $(seq 1 18)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Collections
for bugId in $(seq 1 28)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Compress
for bugId in $(seq 1 47)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Csv
for bugId in $(seq 1 16)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Gson
for bugId in $(seq 1 18)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=JacksonCore
for bugId in $(seq 1 26)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=JacksonDatabind
for bugId in $(seq 1 112)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=JacksonXml
for bugId in $(seq 1 6)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Jsoup
for bugId in $(seq 1 93)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=JxPath
for bugId in $(seq 1 22)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Lang
for bugId in $(seq 1 65)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Math
for bugId in $(seq 1 106)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Mockito
for bugId in $(seq 1 38)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done

proj=Time
for bugId in $(seq 1 27)
do
        defects4j checkout -p $proj -v ${bugId}b -w ${dir}${proj}_${bugId}
done