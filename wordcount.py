from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split
import yaml

with open("spark_config.yaml", "r") as f:
    config = yaml.safe_load(f)

spark = SparkSession.builder \
    .appName(config["spark"]["app"]["name"]) \
    .config("spark.executor.memory", config["spark"]["executor"]["memory"]) \
    .config("spark.executor.cores", config["spark"]["executor"]["cores"]) \
    .master("local[*]") \
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