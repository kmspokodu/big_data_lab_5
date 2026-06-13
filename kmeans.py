from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import yaml

with open("spark_config.yaml", "r") as f:
    config = yaml.safe_load(f)

spark = SparkSession.builder \
    .appName(config["spark"]["app"]["name"]) \
    .config("spark.executor.memory", config["spark"]["executor"]["memory"]) \
    .config("spark.executor.cores", config["spark"]["executor"]["cores"]) \
    .master("local[*]") \
    .getOrCreate()



df = spark.read.csv(
    "en.openfoodfacts.org.products.csv",
    sep="\t",
    header=True,
    inferSchema=True
)

# Выбираю только числовые признаки для корректной кластеризации
features = [
    "energy-kcal_100g",
    "fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "proteins_100g",
    "salt_100g"
]

df = df.select(features)
df = df.dropna()

# Фильтрую выбросы по разумным границам для каждого признака
df = df.filter("""
    `energy-kcal_100g` >= 0 AND `energy-kcal_100g` <= 1000 AND
    fat_100g >= 0 AND fat_100g <= 100 AND
    carbohydrates_100g >= 0 AND carbohydrates_100g <= 100 AND
    sugars_100g >= 0 AND sugars_100g <= 100 AND
    proteins_100g >= 0 AND proteins_100g <= 100 AND
    salt_100g >= 0 AND salt_100g <= 20
""")


df = df.sample(fraction=0.05, seed=42)

print("Final dataset size:", df.count())


assembler = VectorAssembler(
    inputCols=features,
    outputCol="features"
)

df_vec = assembler.transform(df)


scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withStd=True,
    withMean=False
)

scaler_model = scaler.fit(df_vec)
df_scaled = scaler_model.transform(df_vec)

kmeans = KMeans(
    k=5,
    seed=42,
    featuresCol="scaledFeatures"
)

model = kmeans.fit(df_scaled)


predictions = model.transform(df_scaled)

predictions.select("prediction").show(10)


print("\nCluster centers:")
for i, c in enumerate(model.clusterCenters()):
    print(f"Cluster {i}: {c}")


evaluator = ClusteringEvaluator(
    featuresCol="scaledFeatures"
)

silhouette = evaluator.evaluate(predictions)

print("\nSilhouette score:", silhouette)


spark.stop()