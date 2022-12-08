import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, IndexToString}
import org.apache.spark.ml.Pipeline

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").option("sep",";").load("data/bank-full.csv")

//Convert string values into numerical values
val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex")
val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex")
val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutcomeIndex")
val yIndexer = new StringIndexer().setInputCol("y").setOutputCol("yIndex")

val mlpd1 = jobIndexer.fit(data).transform(data)
val mlpd2 = maritalIndexer.fit(mlpd1).transform(mlpd1)
val mlpd3 = educationIndexer.fit(mlpd2).transform(mlpd2)
val mlpd4 = monthIndexer.fit(mlpd3).transform(mlpd3)
val mlpd5 = poutcomeIndexer.fit(mlpd4).transform(mlpd4)
val mlpd = yIndexer.fit(mlpd5).transform(mlpd5)

val assembler = new VectorAssembler().setInputCols(Array("age","jobIndex","maritalIndex","educationIndex","day","monthIndex","duration","campaign","pdays","previous","yIndex","poutcomeIndex")).setOutputCol("features")

val features = assembler.transform(mlpd)

val indexerLabel = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(features)
val indexerFeatures = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(12)

val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)

// ====================================================================
//                         Multilayer Perceptron
// ====================================================================

val layers = Array[Int](12, 5, 4, 2)

val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234).setMaxIter(100)

val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerLabel.labels)

val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))

val model = pipeline.fit(training)

val result = model.transform(test)

val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(result)

println(s"Result of accuracy = ${accuracy}")