import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").option("sep",";").load("data/bank-full.csv")

//Create  numerical "x" column from string "y" column
val cols = data.select("age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y").withColumn("x", when(col("y") === "yes", 1).when(col("y") === "no", 0))

cols.printSchema()

cols.head(1)

val mlpdataall = (cols.select(cols("x").as("label"), $"age", $"job",$"marital", $"education", $"day", $"month", $"duration", $"campaign", $"pdays", $"previous", $"poutcome"))

val mlpdata = mlpdataall.na.drop()


//Convert string values into numerical values
val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex")
val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex")
val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutcomeIndex")

//Convert numerical values into OneHot Encoding 0 -1
val jobEncoder = new OneHotEncoder().setInputCol("jobIndex").setOutputCol("JobVec")
val maritalEncoder = new OneHotEncoder().setInputCol("maritalIndex").setOutputCol("MaritalVec")
val educationalEncoder = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("EducationVec")
val monthEncoder = new OneHotEncoder().setInputCol("monthIndex").setOutputCol("monthVec")
val poutcomeEncoder = new OneHotEncoder().setInputCol("poutcomeIndex").setOutputCol("PoutcomeVec")


val assembler = new VectorAssembler().setInputCols(Array("age","JobVec","MaritalVec","EducationVec","day","monthVec","duration","campaign","pdays","previous","PoutcomeVec")).setOutputCol("features")

// ====================================================================
//                         Multilayer Perceptron
// ====================================================================

val splits = mlpdata.randomSplit(Array(0.7, 0.3), seed = 1234L)

val train = splits(0)
val test = splits(1)

val layers = Array[Int](4, 5, 4, 2)

val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setFeaturesCol("features")
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

val model = trainer.fit(train)

val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")