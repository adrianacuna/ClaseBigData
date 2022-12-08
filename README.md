# ClaseBigData from unit3 branch
## Final Project
### Content menu
+ [Logistic Regression](#Logistic-regression)
+ [Multilayer Perceptron](#multilayer-perceptron)

## Logistic Regression. 
##### Exercises for Final project relative to Logistic regression.

Import libraries that we need for Logistic Regression
```sh
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
```

Create the logger error
```sh
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

```
Create the Spark variable to assing the main session.
```sh
val spark = SparkSession.builder().getOrCreate()
```

**Print Result**
```sh
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@caaf5b
```

Load the file bank-full.cvs that we use to do the following operations and create the model.
```sh
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").option("sep",";").load("data/bank-full.csv")
```

**Print Result**
```sh
data: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
```

Create the columns variable to select the main columns of the main file imported, adding validation to transform the "Yes" and "No" strings to "1" and "0" respective.
```sh 
val cols = data.select("age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y").withColumn("x", when(col("y") === "yes", 1).when(col("y") === "no", 0))
```

**Print Result**
```sh
cols: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
```

See the schema for Cols variable.
```sh 
cols.printSchema()
```

**Print Result**
```sh
root
 |-- age: integer (nullable = true)
 |-- job: string (nullable = true)
 |-- marital: string (nullable = true)
 |-- education: string (nullable = true)
 |-- default: string (nullable = true)
 |-- balance: integer (nullable = true)
 |-- housing: string (nullable = true)
 |-- loan: string (nullable = true)
 |-- contact: string (nullable = true)
 |-- day: integer (nullable = true)
 |-- month: string (nullable = true)
 |-- duration: integer (nullable = true)
 |-- campaign: integer (nullable = true)
 |-- pdays: integer (nullable = true)
 |-- previous: integer (nullable = true)
 |-- poutcome: string (nullable = true)
 |-- y: string (nullable = true)
 |-- x: integer (nullable = true)
 ```


Print the first head value for Cols variable. 
```sh 
cols.head(1)
```

**Print Result**
```sh
res2: Array[org.apache.spark.sql.Row] = Array([58,management,married,tertiary,no,2143,yes,no,unknown,5,may,261,1,-1,0,unknown,no,0])
```

Adding metadata for label column assigned to logregdatall variable
```sh 
val logregdatall = (cols.select(cols("x").as("label"), $"age", $"job",
                    $"marital", $"education", $"day", $"month", $"duration", $"campaign", $"pdays", $"previous", $"poutcome"))
```

**Print Result**
```sh
logregdatall: org.apache.spark.sql.DataFrame = [label: int, age: int ... 10 more fields]
```

Create a new dataframe dropping all missing values for logregdata value. 
```sh 
val logregdata = logregdatall.na.drop()
```

**Print Result**
```sh
logregdata: org.apache.spark.sql.DataFrame = [label: int, age: int ... 10 more fields]
```

Convert string values into numerical values
```sh 
val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex")
val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex")
val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutcomeIndex")
```

**Print Result**
```sh
jobIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_70ef7df84d44
maritalIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_978b1053e542
educationIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_88c70d3622b7
monthIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_41a39f69635e
poutcomeIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_0eb3209022d4
```


Convert numerical values into OneHot Encoding 0 -1
```sh 
val jobEncoder = new OneHotEncoder().setInputCol("jobIndex").setOutputCol("JobVec")
val maritalEncoder = new OneHotEncoder().setInputCol("maritalIndex").setOutputCol("MaritalVec")
val educationalEncoder = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("EducationVec")
val monthEncoder = new OneHotEncoder().setInputCol("monthIndex").setOutputCol("monthVec")
val poutcomeEncoder = new OneHotEncoder().setInputCol("poutcomeIndex").setOutputCol("PoutcomeVec")
```

**Print Result**
```sh
jobEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_f08ed954a99c
maritalEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_a072f93032be
educationalEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_bea0bdc61190
monthEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_0a73b592b288
poutcomeEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_5c1f2466e00a
```

We have a dataframe has the column headings, we need a column called “label” and one called “features” for LR algorithm. So we use the VectorAssembler() to do that.
The label indicated whether the comsumer has a term deposit plan..
```sh 
val assembler = (new VectorAssembler()
                  .setInputCols(Array("age","JobVec","MaritalVec","EducationVec","day","monthVec","duration","campaign","pdays","previous","PoutcomeVec"))
                  .setOutputCol("features"))
```

**Print Result**
```sh
assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_7cf90f3ff6b8, handleInvalid=error, numInputCols=11
```

We splitting the data randomly to generate two sets: one to use during training of the ML algorithm (training set), and the second to check whether the training is working (test set).
```sh
val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
```

**Print Result**
```sh
training: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, age: int ... 10 more fields]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, age: int ... 10 more fields]
```

Declare the logistic regression method for used to predict the binary response.
```sh 
val lr = new LogisticRegression()
```

**Print Result**
```sh
lr: org.apache.spark.ml.classification.LogisticRegression = logreg_11004705b066
```


Declare a pipeline to use multiple transformers and estimators together to specify the machine learning workflow.
```sh 
val pipeline = new Pipeline().setStages(Array(jobIndexer,maritalIndexer,educationIndexer,monthIndexer,poutcomeIndexer,
                    jobEncoder,maritalEncoder,educationalEncoder,monthEncoder,poutcomeEncoder,assembler,lr))
```

**Print Result**
```sh
pipeline: org.apache.spark.ml.Pipeline = pipeline_9758bce532b8
```

For estimator stages, the fit() method is called to produce a Transformer.
```sh 
val model = pipeline.fit(training)
```

**Print Result**
```sh
model: org.apache.spark.ml.PipelineModel = pipeline_9758bce532b8
```

We make predictions on test data using the transform() method.
```sh 
val results = model.transform(test)
```

**Print Result**
```sh
results: org.apache.spark.sql.DataFrame = [label: int, age: int ... 24 more fields]
```

Model testing
```sh 
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
```

**Print Result**
```sh
predictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[137] at rdd at <console>:33
```

```sh
val metrics = new MulticlassMetrics(predictionAndLabels)
```

**Print Result**
```sh
metrics: org.apache.spark.mllib.evaluation.MulticlassMetrics = org.apache.spark.mllib.evaluation.MulticlassMetrics@408d8a13
```

 Sets the value of metricLabel for MulticlassClassificationEvaluator
```sh 
val evaluador = new MulticlassClassificationEvaluator().setMetricName("accuracy")
```

**Print Result**
```sh
evaluador: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_8e9fed779a79, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15
```

Print out the Confusion matrix
```sh 
println("Confusion matrix:")
println(metrics.confusionMatrix)
```

**Print Result**
```sh
scala> println(metrics.confusionMatrix)
11862.0  279.0  
1000.0   511.0
```

Print out the accuracy
println(s"accuracy = ${evaluador.evaluate(results)}")
```

**Print Result**
```sh
accuracy = 0.9063140931731615
```

## Multilayer Perceptron
##### Exercises for eFinal Project relative to Machine Learning Multilayer Perceptron.

Import libraries needed for  Multilayer Perceptron in Spark.
```sh
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}

    import org.apache.log4j._
    
```
Create the logger error 
```sh
    Logger.getLogger("org").setLevel(Level.ERROR)
```

Create the Spark variable to assing the main session. 
```sh
    val spark = SparkSession.builder().getOrCreate()
```
**Print Result**
```sh
    scala> val spark = SparkSession.builder().getOrCreate()
    spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@5126fb1d
```

Load the file bank-full.cvs that we use to do the following operations and create the model.
```sh
    val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").option("sep",";").load("data/bank-full.csv")
```
**Print Result**
```sh
    scala> val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").option("sep",";").load("data/bank-full.csv")
    data: org.apache.spark.sql.DataFrame = [age: int, job: string ... 15 more fields]
```

See the schema for Data Set.
```sh
    data.printSchema()
```
**Print Result**
```sh
    scala> data.printSchema()
    root
    |-- age: integer (nullable = true)
    |-- job: string (nullable = true)
    |-- marital: string (nullable = true)
    |-- education: string (nullable = true)
    |-- default: string (nullable = true)
    |-- balance: integer (nullable = true)
    |-- housing: string (nullable = true)
    |-- loan: string (nullable = true)
    |-- contact: string (nullable = true)
    |-- day: integer (nullable = true)
    |-- month: string (nullable = true)
    |-- duration: integer (nullable = true)
    |-- campaign: integer (nullable = true)
    |-- pdays: integer (nullable = true)
    |-- previous: integer (nullable = true)
    |-- poutcome: string (nullable = true)
    |-- y: string (nullable = true)
```
Print the first head value for Cols variable. 
```sh
    data.head(1)
```
**Print Result**
```sh
    scala> data.head(1)
    res72: Array[org.apache.spark.sql.Row] = Array([58,management,married,tertiary,no,2143,yes,no,unknown,5,may,261,1,-1,0,unknown,no])
```
See the first 5 row of data 
```sh
    data.show(5)
```
**Print Result**
```sh
    scala> data.show(5)
    +---+------------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
    |age|         job|marital|education|default|balance|housing|loan|contact|day|month|duration|campaign|pdays|previous|poutcome|  y|
    +---+------------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
    | 58|  management|married| tertiary|     no|   2143|    yes|  no|unknown|  5|  may|     261|       1|   -1|       0| unknown| no|
    | 44|  technician| single|secondary|     no|     29|    yes|  no|unknown|  5|  may|     151|       1|   -1|       0| unknown| no|
    | 33|entrepreneur|married|secondary|     no|      2|    yes| yes|unknown|  5|  may|      76|       1|   -1|       0| unknown| no|
    | 47| blue-collar|married|  unknown|     no|   1506|    yes|  no|unknown|  5|  may|      92|       1|   -1|       0| unknown| no|
    | 33|     unknown| single|  unknown|     no|      1|     no|  no|unknown|  5|  may|     198|       1|   -1|       0| unknown| no|
    +---+------------+-------+---------+-------+-------+-------+----+-------+---+-----+--------+--------+-----+--------+--------+---+
    only showing top 5 rows
```
Describe function we can see the mean, standard deviation, and minimum and maximum value for each numerical column
```sh
    data.describe().show()
```
**Print Result**
```sh
    scala> data.describe().show()
    +-------+------------------+-------+--------+---------+-------+------------------+-------+-----+--------+-----------------+-----+-----------------+------------------+------------------+------------------+--------+-----+
    |summary|               age|    job| marital|education|default|           balance|housing| loan| contact|              day|month|         duration|          campaign|             pdays|          previous|poutcome|    y|
    +-------+------------------+-------+--------+---------+-------+------------------+-------+-----+--------+-----------------+-----+-----------------+------------------+------------------+------------------+--------+-----+
    |  count|             45211|  45211|   45211|    45211|  45211|             45211|  45211|45211|   45211|            45211|45211|            45211|             45211|             45211|             45211|   45211|45211|
    |   mean| 40.93621021432837|   null|    null|     null|   null|1362.2720576850766|   null| null|    null|15.80641879188693| null|258.1630797814691| 2.763840658246887| 40.19782796222158|0.5803233726305546|    null| null|
    | stddev|10.618762040975401|   null|    null|     null|   null|3044.7658291685243|   null| null|    null|8.322476153044589| null|257.5278122651712|3.0980208832791813|100.12874599059818|2.3034410449312164|    null| null|
    |    min|                18| admin.|divorced|  primary|     no|             -8019|     no|   no|cellular|                1|  apr|                0|                 1|                -1|                 0| failure|   no|
    |    max|                95|unknown|  single|  unknown|    yes|            102127|    yes|  yes| unknown|               31|  sep|             4918|                63|               871|               275| unknown|  yes|
```

Convert string columns values to numerical (integer) values. 
```sh
    val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
    val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex")
    val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
    val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex")
    val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutcomeIndex")
    val yIndexer = new StringIndexer().setInputCol("y").setOutputCol("yIndex")
```
**Print Result**
```sh
    scala> val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
    jobIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_c54f9e37d8f1

    scala> val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex")
    maritalIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_be889ddfe22d

    scala> val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
    educationIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_dd3b4278ac22

    scala> val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex")
    monthIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_177bfddb27b7

    scala> val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutcomeIndex")
    poutcomeIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_f2312c84b89e

    scala> val yIndexer = new StringIndexer().setInputCol("y").setOutputCol("yIndex")
    yIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_ff448dc40e07
```
Merge the values converted into numerical values to the main data, adding the next merged data before to the new ones.
```sh
    val mlpd1 = jobIndexer.fit(data).transform(data)
    val mlpd2 = maritalIndexer.fit(mlpd1).transform(mlpd1)
    val mlpd3 = educationIndexer.fit(mlpd2).transform(mlpd2)
    val mlpd4 = monthIndexer.fit(mlpd3).transform(mlpd3)
    val mlpd5 = poutcomeIndexer.fit(mlpd4).transform(mlpd4)
    val mlpd = yIndexer.fit(mlpd5).transform(mlpd5)
```
**Print Result**
```sh
   scala> val mlpd1 = jobIndexer.fit(data).transform(data)
    mlpd1: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]

    scala> val mlpd2 = maritalIndexer.fit(mlpd1).transform(mlpd1)
    mlpd2: org.apache.spark.sql.DataFrame = [age: int, job: string ... 17 more fields]

    scala> val mlpd3 = educationIndexer.fit(mlpd2).transform(mlpd2)
    mlpd3: org.apache.spark.sql.DataFrame = [age: int, job: string ... 18 more fields]

    scala> val mlpd4 = monthIndexer.fit(mlpd3).transform(mlpd3)
    mlpd4: org.apache.spark.sql.DataFrame = [age: int, job: string ... 19 more fields]

    scala> val mlpd5 = poutcomeIndexer.fit(mlpd4).transform(mlpd4)
    mlpd5: org.apache.spark.sql.DataFrame = [age: int, job: string ... 20 more fields]

    scala> val mlpd = yIndexer.fit(mlpd5).transform(mlpd5)
    mlpd: org.apache.spark.sql.DataFrame = [age: int, job: string ... 21 more fields]
```
Merge multiple columns into a vector column named Features with VectorAssembler method.
```sh
    val assembler = new VectorAssembler().setInputCols(Array("age","jobIndex","maritalIndex","educationIndex","day","monthIndex","duration","campaign","pdays","previous","yIndex","poutcomeIndex")).setOutputCol("features")
```
**Print Result**
```sh
    scala> val assembler = new VectorAssembler().setInputCols(Array("age","jobIndex","maritalIndex","educationIndex","day","monthIndex","duration","campaign","pdays","previous","yIndex","poutcomeIndex")).setOutputCol("features")
    assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_3bd3df209038, handleInvalid=error, numInputCols=12
```
Transfor the assembler with the data merged and assign to features variable 
```sh
    val features = assembler.transform(mlpd)
```
**Print Result**
```sh
    scala> val features = assembler.transform(mlpd)
    features: org.apache.spark.sql.DataFrame = [age: int, job: string ... 22 more fields]
```
Create new stringIndexer of Y column to create new column merging features.
```sh
    val indexerLabel = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(features) 
```
**Print Result**
```sh
    scala> val indexerLabel = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(features)
    indexerLabel: org.apache.spark.ml.feature.StringIndexerModel = StringIndexerModel: uid=strIdx_110c0fe82aee, handleInvalid=error
```
Create new VectorIndexer of features column to create new indexed features column defining the number of categories.
```sh
   val indexerFeatures = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(12)
```
**Print Result**
```sh
    scala> val indexerFeatures = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(12)
    indexerFeatures: org.apache.spark.ml.feature.VectorIndexer = vecIdx_52e7368db34e
```
Split the data, using some of it to train the model and reserving some to test the trained model. We use 70% of the data for training, and 30% for testing.
```sh
    val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)
```
**Print Result**
```sh
    scala> val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)
    training: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [age: int, job: string ... 22 more fields]
    test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [age: int, job: string ... 22 more fields]
```
Define the value layers that contains the array refering to the inputs layers (the inputs columns ) the hide layers and the output layer (number of species).
```sh
    val layers = Array[Int](12, 5, 4, 2)
```
**Print Result**
```sh
    scala> val layers = Array[Int](12, 5, 4, 2)
    layers: Array[Int] = Array(12, 5, 4, 2)
```
Set the new classifier and set layers, define label colums, features and set block size, seed and max inter. 
```sh
   val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234).setMaxIter(100)
```
**Print Result**
```sh
    scala> val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234).setMaxIter(100)
    trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_57e285ad2b5a
```
Create new IndexToString to set input column called "predictedLabel", and convert indexed labels back to original labels
```sh
   val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerLabel.labels)
```
**Print Result**
```sh
    scala> val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerLabel.labels)
    warning: one deprecation (since 3.0.0); for details, enable `:setting -deprecation' or `:replay -deprecation'
    converterLabel: org.apache.spark.ml.feature.IndexToString = idxToStr_4810dbf8f695
```
Create new Pipeline with the following elements: indexerLabel, indexerFeatures, trainer, converterLabel. 
```sh
    val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))
```
**Print Result**
```sh
    scala> val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))
    pipeline: org.apache.spark.ml.Pipeline = pipeline_6bf2fda0223c
```
Train the model using the pipeline created, using fit to transform. 
```sh
   val model = pipeline.fit(training)
    
```
**Print Result**
```sh
    scala> val model = pipeline.fit(training)
    model: org.apache.spark.ml.PipelineModel = pipeline_6bf2fda0223c
```
Converts the DataFrame with the test results
```sh
    val result = model.transform(test)
```
**Print Result**
```sh
    scala> val result = model.transform(test)
    result: org.apache.spark.sql.DataFrame = [age: int, job: string ... 28 more fields]
```
Define the parameters for the labels, prediction column and set the metric name into evaluator variable using Multiclassfictation evaluator.
```sh
   val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
```
**Print Result**
```sh
    scala> val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_4b5972b50b52, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15
```
Evaluate the accuracy using the results data from evaluator variable.
```sh
    val accuracy = evaluator.evaluate(result)
```
**Print Result**
```sh
    scala> val accuracy = evaluator.evaluate(result)
    accuracy: Double = 0.8796513331380018
```
Print the accuracy and the error values.
```sh
    println(s"Result of accuracy = ${accuracy}") 
```
**Print Result**
```sh
    scala> println(s"Result of accuracy = ${accuracy}")
    Result of accuracy = 0.8796513331380018
```