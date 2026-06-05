from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split

spark = SparkSession.builder \
    .appName("WordCount") \
    .getOrCreate()

# читаем текст
df = spark.read.text("input.txt")

# разбиваем на слова
words = df.select(explode(split(df.value, " ")).alias("word"))

# считаем
word_counts = words.groupBy("word").count()

# вывод
word_counts.show()

spark.stop()