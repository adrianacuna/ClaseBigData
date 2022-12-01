
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder,IndexToString, StringIndexer}

import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").option("sep",";").load("data/bank-full.csv")

data.printSchema()
data.show(1)

//val col = ("default", "housing", "loan", "y")

val depdata = data.select(data("y").as("label"), $"default", $"age", $"housing", $"loan")

val cleanData = depdata.na.drop()

val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("labelIndex")
val defaultIndexer = new StringIndexer().setInputCol("default").setOutputCol("defaultIndex")
val housingIndexer = new StringIndexer().setInputCol("housing").setOutputCol("housingIndex")
val loanIndexer = new StringIndexer().setInputCol("loan").setOutputCol("loanIndex")

val labelEncoder = new OneHotEncoder().setInputCol("labelIndex").setOutputCol("labelEnc")
val defaultEncoder = new OneHotEncoder().setInputCol("defaultIndex").setOutputCol("defaultEnc")
val housingEncoder = new OneHotEncoder().setInputCol("housingIndex").setOutputCol("housingEnc")
val loanEncoder = new OneHotEncoder().setInputCol("loanIndex").setOutputCol("loanEnc")

val assembler = (new VectorAssembler().setInputCols(Array("labelEnc", "defaultEnc","age", "housingEnc","loanEnc")).setOutputCol("features"))

///////////////////////////////
// Logistic Regression ///////
//////////////////////////////

val Array(training, test) = cleanData.randomSplit(Array(0.7, 0.3), seed = 12345)

val lr = new LogisticRegression()

val pipeline = new Pipeline().setStages(Array(labelIndexer, defaultIndexer, housingIndexer, loanIndexer, labelEncoder, defaultEncoder, housingEncoder, loanEncoder, assembler, lr))

val model = pipeline.fit(training)

val results = model.transform(test)

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy



///////////////////////////////
// Multilayer perceptron ///////
//////////////////////////////

val splits = cleanData.randomSplit(Array(0.7, 0.3), seed = 1234)
val train = splits(0)
val test = splits(1)

val layers = Array[Int](4, 5, 4, 2)

val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

  val model = trainer.fit(train)


val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")