# ClaseBigData from unit2 branch
### Content menu
+ [Practice 1](#practice-1-linear-regression)
+ [Practice 2](#practice-2-logistic-regression)
+ [Practice 3](#practice-3-decision-tree)
+ [Practice 4](#practice-4-random-forest)
+ [Practice 5](#practice-5-multilayer-peceptron-classifier)
+ [Evaluation Unit 2](#evaluation-unit-2)

## Practice 1. Linear Regression. 
##### Practice to resolve and test Linear Regression methods from AssigmentLinearRegression.scala file.

Import LinearRegression
```sh
import org.apache.spark.ml.regression.LinearRegression
```

Opcional: Utilice el siguiente codigo para configurar errores
```sh
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)
```

Start a simple Spark Session
```sh
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
```

Use Spark to read the Ecommerce Customers csv file.
```sh
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")
```

Print Schema of the DataFrame
```sh
data.printSchema
```
**Print result**
```sh
scala> data.printSchema
root
 |-- Email: string (nullable = true)
 |-- Avatar: string (nullable = true)
 |-- Avg Session Length: double (nullable = true)
 |-- Time on App: double (nullable = true)
 |-- Time on Website: double (nullable = true)
 |-- Length of Membership: double (nullable = true)
 |-- Yearly Amount Spent: double (nullable = true)
 ```

Imprima un renglon de ejemplo del DataFrane.
Print one example row of the dataframe

```sh
data.head(1)
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example data row")
for(ind <- Range(1, colnames.length)){
    println(colnames(ind))
    println(firstrow(ind))
    println("\n")
}
```
**Print result**
```sh
scala> println("Example data row")
Example data row

scala> for(ind <- Range(1, colnames.length)){
     |     println(colnames(ind))
     |     println(firstrow(ind))
     |     println("\n")
     | }
Avatar
Violet


Avg Session Length
34.49726772511229


Time on App
12.65565114916675


Time on Website
39.57766801952616


Length of Membership
4.0826206329529615


Yearly Amount Spent
587.9510539684005

```

##### Configure el DataFrame para Machine Learning


Transform the dataframe to  take the for of ("label","features")
Import VectorAssembler y Vectors

```sh
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```

Rename the Yearly Amount Spent Column as "label", also grab only the numerical columns from the data
Set all of this as a new dataframe called df

```sh
val df = data.select(data("Yearly Amount Spent").as("label"),
$"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")
```
**Print result**
```sh
df: org.apache.spark.sql.DataFrame = [label: double, Avg Session Length: double ... 3 more fields]
```

An assembler converts the input values to a vector

Use VectorAssembler to convert the input columns of df to a single output column of an array called "features"
Set the input columns from which we are supposed to read the values.
Call this new object assembler

```sh
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")
```
**Print result**
```sh
assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_b3c45a0cf4e8, handleInvalid=error, numInputCols=4
```

Use the assembler to transform the DataFrame to the two columns: label and features

```sh
val output = assembler.transform(df).select($"label",$"features")
```
**Print result**
```sh
output: org.apache.spark.sql.DataFrame = [label: double, features: vector]
```

Create a Linear Regression Model object

```sh
val lr = new LinearRegression()
```
**Print result**
```sh
lr: org.apache.spark.ml.regression.LinearRegression = linReg_022cc6b7f4ca
```

Fit the model to the data and call this model lrModel

```sh
val lrModelo = lr.fit(output)
```
**Print result**
```sh
lrModelo: org.apache.spark.ml.regression.LinearRegressionModel = LinearRegressionModel: uid=linReg_022cc6b7f4ca, numFeatures=4
```

Print the coefficients and intercept for linear regression

```sh
println(s"Coefficients: ${lrModelo.coefficients} Intercept: ${lrModelo.intercept}")
```
**Print result**
```sh
Coefficients: [25.734271084670716,38.709153810828816,0.43673883558514964,61.57732375487594] Intercept: -1051.5942552990748
```

Summarize the model over the training set and print out some metrics!
Use the .summary method off your model to create an object called trainingSummary

```sh
val trainingSummary = lrModelo.summary
```

**Print result**
```sh
trainingSummary: org.apache.spark.ml.regression.LinearRegressionTrainingSummary = org.apache.spark.ml.regression.LinearRegressionTrainingSummary@7d026083
```

Show the residuals, the RMSE, the MSE, and the R^2 Values.

```sh
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
```
**Print result**
```sh
scala> trainingSummary.residuals.show()
+-------------------+
|          residuals|
+-------------------+
| -6.788234090018818|
| 11.841128565326073|
| -17.65262700858966|
| 11.454889631178617|
| 7.7833824373080915|
|-1.8347332184773677|
|  4.620232401352382|
| -8.526545950978175|
| 11.012210896516763|
|-13.828032682158891|
| -16.04456458615175|
|  8.786634365463442|
| 10.425717191807507|
| 12.161293785003522|
|  9.989313714461446|
| 10.626662732649379|
|  20.15641408428496|
|-3.7708446586326545|
| -4.129505481591934|
|  9.206694655890487|
+-------------------+
only showing top 20 rows


scala> println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
RMSE: 9.923256785022229

scala> println(s"MSE: ${trainingSummary.meanSquaredError}")
MSE: 98.47102522148971

scala> println(s"r2: ${trainingSummary.r2}")
r2: 0.9843155370226727
    
```

## Practice 2. Logistic Regression.
##### Practice to analyze and document the PracticalLogisticRegression.scala file.

Import the SparkSession using the Logistic Regression library
```sh
    import org.apache.spark.ml.classification.LogisticRegression
```
Optional: Use the Error reporting code
```sh
    import org.apache.log4j._
    Logger.getLogger("org").setLevel(Level.ERROR)
```
Create the basic SparkSession and build the variable spark
```sh
    import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder().getOrCreate()
```
Read the Advertising csv file using Spark and prepare to work
```sh
    val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
```
**Print Data**
Print the DataFrame schema
```sh
    data.printSchema()
```
**Print result**
```sh
    scala> data.printSchema()
    root
    |-- Daily Time Spent on Site: double (nullable = true)
    |-- Age: integer (nullable = true)
    |-- Area Income: double (nullable = true)
    |-- Daily Internet Usage: double (nullable = true)
    |-- Ad Topic Line: string (nullable = true)
    |-- City: string (nullable = true)
    |-- Male: integer (nullable = true)
    |-- Country: string (nullable = true)
    |-- Timestamp: timestamp (nullable = true)
    |-- Clicked on Ad: integer (nullable = true)
```
Prit a row example of data
```sh
    data.head(1)

    val colnames = data.columns
    val firstrow = data.head(1)(0)
    println("\n")
    println("Example data row")
    for(ind <- Range(1, colnames.length)){
        println(colnames(ind))
        println(firstrow(ind))
        println("\n")
    }
```
**Print result**
```sh
    scala> println("\n")

    scala> println("Example data row")
    Example data row

    scala> for(ind <- Range(1, colnames.length)){
        |     println(colnames(ind))
        |     println(firstrow(ind))
        |     println("\n")
        | }
    Age
    35


    Area Income
    61833.9


    Daily Internet Usage
    256.09


    Ad Topic Line
    Cloned 5thgeneration orchestration


    City
    Wrightburgh


    Male
    0


    Country
    Tunisia


    Timestamp
    2016-03-27 00:53:11.0


    Clicked on Ad
    0
```
**Prepare DataFrame for Machine Learning**

Rename the column "Clicked on Ad" to "label", consider the feeatures colums such a Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"; and create new column named "Hour" from the Timestamp content "Hour of the click".
```sh
    val timedata = data.withColumn("Hour",hour(data("Timestamp")))

    val logregdata = timedata.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", $"Hour", $"Male")
```
**Print result**
```sh
    timedata: org.apache.spark.sql.DataFrame = [Daily Time Spent on Site: double, Age: int ... 9 more fields]
    logregdata: org.apache.spark.sql.DataFrame = [label: int, Daily Time Spent on Site: double ... 5 more fields]
```
Import the VectorAssembler and import Vectors too.
```sh
    import org.apache.spark.ml.feature.VectorAssembler
    import org.apache.spark.ml.linalg.Vectors
```

Create new VectorAssembler object called "assembler" to use in features.
```sh
    val assembler = (new VectorAssembler()
                    .setInputCols(Array("Daily Time Spent on Site", "Age","Area Income","Daily Internet Usage","Hour","Male"))
                    .setOutputCol("features"))
```
**Print result**
```sh
    assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_f321d86871ce, handleInvalid=error, numInputCols=6
```

Use randomSplit to create data of train and testing with of 70/30 dividers
```sh
    val Array(training, test) = logregdata.randomSplit(Array(0.7, 0.3), seed = 12345)
```
**Print result**
```sh
    training: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, Daily Time Spent on Site: double ... 5 more fields]

    test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, Daily Time Spent on Site: double ... 5 more fields]
```
**Pipeline Configuration**

Import the Pipeline 
```sh
    import org.apache.spark.ml.Pipeline
```
Create new LogisticRegression object and called "lr"
```sh
    val lr = new LogisticRegression()
```
**Print result**
```sh
    lr: org.apache.spark.ml.classification.LogisticRegression = logreg_e50a6bf3435b
```
Create new Pipeline with the followign elements: "assembler" and "lr"
```sh
    val pipeline = new Pipeline().setStages(Array(assembler, lr))
```
**Print result**
```sh
    pipeline: org.apache.spark.ml.Pipeline = pipeline_9a0f1dc04d31
```
Fit the Pipeline for the training conjunction. 
```sh
    val model = pipeline.fit(training)
```
**Print result**
```sh
    model: org.apache.spark.ml.PipelineModel = pipeline_9a0f1dc04d31
```
Get the result using the conjunction for transformation test
```sh
    val results = model.transform(test)
```
**Print result**
```sh
    results: org.apache.spark.sql.DataFrame = [label: int, Daily Time Spent on Site: double ... 9 more fields]
```

**Model Evaluation**

For the Metrics and Evaluation import MulticlassMetrics
```sh
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
```
Convert the test results on RDD using ".as" and ".rdd"
```sh
    val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)
```
**Print result**
```sh
    predictionAndLabels: org.apache.spark.rdd.RDD[(Double, Double)] = MapPartitionsRDD[68] at rdd at <console>:32

    metrics: org.apache.spark.mllib.evaluation.MulticlassMetrics = org.apache.spark.mllib.evaluation.MulticlassMetrics@71b4b729
```
Initialize the MulticlassMetrics object
```sh
    println("Confusion matrix:")
    println(metrics.confusionMatrix)
```
**Print result**
```sh
    scala> println("Confusion matrix:")
    Confusion matrix:

    scala> println(metrics.confusionMatrix)
    136.0  1.0    
    4.0    146.0  
```
Print the Confusion matrix
```sh
    metrics.accuracy
```
**Print result**
```sh
    res8: Double = 0.9825783972125436
```

## Practice 3. Decision Tree. 
##### Practice to run and document our observations of the example of the Spark documentation for Decision Tree Classifier.


Import SpakConf and SparkContext that set common properties for the application.
```sh
import org.apache.spark.{SparkConf, SparkContext}
```
Import machine learning library to use DescisionTree  methods like classification and regression.
```sh
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
```

Import MLUtils to use helper methods to load, save and process data used in Machine Learning methods.
```sh
import org.apache.spark.mllib.util.MLUtils
```
Create a new object named DecisionTreeClassification that containt all the operartion and function for this example.
```sh
object DecisionTreeClassificationExample {
```
Create the main class  whit a array argument, create new variable conf that contains the default app settings where we define the name of the app. And load that settings whit the method SparkContext.
```sh
def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("DecisionTreeClassificationExample")
    val sc = new SparkContext(conf)
```

Used the helper methods to load the data that we used for this example for MachineLearning
```sh
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
```

We used the method randomSplit to split the dataframe into two a specific weights, in this case 70% and 30 %, and assign the result into training Data y testData variables.
```sh
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
```

Next was define the values for the number of classes than refers the numbers of outcomes for this model. 
```sh
val numClasses = 2
```

Also specified that the categorical features are continuous.
val categoricalFeaturesInfo = Map[Int, Int]()
```

A Impurity value mesuare = gini is used to choose between candidate splits.
```sh
val impurity = "gini"
```

This is is the training paranmeter that define the node depth.
```sh 
val maxDepth = 5
```

The value maxBins refers to number of bins used when discretizing continuos features.
```sh
val maxBins = 32
```

Next we use the method to train a decision tree model.
```sh
val model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,impurity, maxDepth, maxBins)
```


Create the values to evaluate the model fot label and predictions.
```sh
val labelAndPreds = testData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
}
```

Following we create the values to calculate the error from the model testing
```sh
val testErr = labelAndPreds.filter(r => r._1 != r._2).count().toDouble / testData.count()
    println(s"Test Error = $testErr")
    println(s"Learned classification tree model:\n ${model.toDebugString}")
```

Save the model to be used later and load the model.
```sh
model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
    val sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")
```

Finally stop all the sparkcontext.
```sh
sc.stop()
```
**Print result**
```sh
defined object DecisionTreeClassificationExample
```
## Practice 4. Random Forest.
##### Practice to run and document our observations of the example of the Spark documentation for Random Forest.

Import the Pipeline
```sh
    import org.apache.spark.ml.Pipeline
```
Import the RandomForestClassificationModel and RandomForestClassifier
```sh
    import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
```
Import MulticlassClassificationEvaluator
```sh
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
```
Import features IndexToString, StringIndexer, and VectorIndexer
```sh
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
```
Import the basic SpakSession 
```sh
    import org.apache.spark.sql.SparkSession
```
Create SparkSession builder for the appaName "RandomForestClassifierExample" and asign to "spark" variable
```sh
    val spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()
```
**Print result**
```sh
    spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@24abee74
```
Read the sample_libsvm_data txt file using Spark and prepare to work
```sh
    val data = spark.read.format("libsvm").load("sample_libsvm_data.txt")
```
**Print result**
```sh
    data: org.apache.spark.sql.DataFrame = [label: double, features: vector]
```
Use StringIndexer to create an input column from label to output column called "indexedLabel" and fit on whole dataset to include all labels. 
```sh
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
```
**Print result**
```sh
    labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = StringIndexerModel: uid=strIdx_f0427f2102bc, handleInvalid=error
```
Use VectorIndexer to create an input column from features to output column called "indexedFeatures", fit on whole dataset to include all labels and set maxCategories to 4. 
```sh
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
```
**Print result**
```sh
    featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = VectorIndexerModel: uid=vecIdx_60cfa85e01df, numFeatures=692, handleInvalid=error
```

Split the data into training and test sets (70/30) using randomSplit
```sh
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
```
**Print result**
```sh
    trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

    testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
```
Use RandomForestClassifier to create an label column of indexedLabel using "indexedFeatures" and setNumTrees to 10 for training a RandomForest model
```sh
    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
```
**Print result**
```sh
    rf: org.apache.spark.ml.classification.RandomForestClassifier = rfc_1b277030fdb0
```
Create new IndexToString to set input column from predictions to output column called "predictedLabel", and convert indexed labels back to original labels
```sh
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labelsArray(0))
```
**Print result**
```sh
    labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_06a4a8385500
```
Create new Pipeline with the following elements: labelIndexer, featureIndexer, rf, and labelConverter
```sh
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
```
**Print result**
```sh
    pipeline: org.apache.spark.ml.Pipeline = pipeline_5a5455d659f6
```
Fit the Pipeline for the training conjunction.
```sh
    val model = pipeline.fit(trainingData)
```
**Print result**
```sh
    model: org.apache.spark.ml.PipelineModel = pipeline_5a5455d659f6
```
Get the result using the conjunction for transformation test
```sh
    val predictions = model.transform(testData)
```
**Print result**
```sh
    predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]
```
Select 5 example rows to display "predictedLabel", "label", "features"
```sh
    predictions.select("predictedLabel", "label", "features").show(5)
```
**Print result**
```sh
    scala> predictions.select("predictedLabel", "label", "features").show(5)
    +--------------+-----+--------------------+
    |predictedLabel|label|            features|
    +--------------+-----+--------------------+
    |           0.0|  0.0|(692,[122,123,148...|
    |           0.0|  0.0|(692,[123,124,125...|
    |           0.0|  0.0|(692,[124,125,126...|
    |           0.0|  0.0|(692,[126,127,128...|
    |           0.0|  0.0|(692,[126,127,128...|
    +--------------+-----+--------------------+
    only showing top 5 rows
```
Create new MulticlassClassificationEvaluator and select (prediction, true label) and compute test error
```sh
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
```
**Print result**
```sh
    scala>     val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_c39325ddd9dc, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15

    scala>     
        |     val accuracy = evaluator.evaluate(predictions)
    accuracy: Double = 1.0

    scala>     println(s"Test Error = ${(1.0 - accuracy)}")
    Test Error = 0.0
```
Print the learned classification forest model
```sh
    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
```
**Print result**
```sh
    scala>     println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
Learned classification forest model:
 RandomForestClassificationModel: uid=rfc_1b277030fdb0, numTrees=10, numClasses=2, numFeatures=692
  Tree 0 (weight 1.0):
    If (feature 455 <= 14.5)
     If (feature 467 <= 70.0)
      Predict: 0.0
     Else (feature 467 > 70.0)
      Predict: 1.0
    Else (feature 455 > 14.5)
     Predict: 1.0
  Tree 1 (weight 1.0):
    If (feature 541 <= 121.5)
     If (feature 384 <= 18.0)
      Predict: 0.0
     Else (feature 384 > 18.0)
      Predict: 1.0
    Else (feature 541 > 121.5)
     If (feature 455 <= 14.5)
      If (feature 432 <= 216.5)
       Predict: 1.0
      Else (feature 432 > 216.5)
       Predict: 0.0
     Else (feature 455 > 14.5)
      Predict: 1.0
  Tree 2 (weight 1.0):
    If (feature 463 <= 2.0)
     If (feature 518 <= 119.5)
      If (feature 432 <= 4.0)
       Predict: 1.0
      Else (feature 432 > 4.0)
       Predict: 0.0
     Else (feature 518 > 119.5)
      Predict: 0.0
    Else (feature 463 > 2.0)
     Predict: 0.0
  Tree 3 (weight 1.0):
    If (feature 216 <= 44.0)
     If (feature 397 <= 2.5)
      If (feature 401 <= 23.5)
       Predict: 0.0
      Else (feature 401 > 23.5)
       Predict: 1.0
     Else (feature 397 > 2.5)
      Predict: 1.0
    Else (feature 216 > 44.0)
     If (feature 345 <= 18.5)
      Predict: 0.0
     Else (feature 345 > 18.5)
      If (feature 468 <= 2.5)
       Predict: 0.0
      Else (feature 468 > 2.5)
       Predict: 1.0
  Tree 4 (weight 1.0):
    If (feature 433 <= 52.5)
     Predict: 1.0
    Else (feature 433 > 52.5)
     Predict: 0.0
  Tree 5 (weight 1.0):
    If (feature 539 <= 33.0)
     If (feature 327 <= 81.0)
      Predict: 0.0
     Else (feature 327 > 81.0)
      Predict: 1.0
    Else (feature 539 > 33.0)
     Predict: 1.0
  Tree 6 (weight 1.0):
    If (feature 511 <= 1.5)
     If (feature 299 <= 214.5)
      Predict: 0.0
     Else (feature 299 > 214.5)
      Predict: 1.0
    Else (feature 511 > 1.5)
     Predict: 1.0
  Tree 7 (weight 1.0):
    If (feature 317 <= 8.0)
     If (feature 244 <= 5.0)
      Predict: 0.0
     Else (feature 244 > 5.0)
      Predict: 1.0
    Else (feature 317 > 8.0)
     If (feature 511 <= 1.5)
      Predict: 0.0
     Else (feature 511 > 1.5)
      Predict: 1.0
  Tree 8 (weight 1.0):
    If (feature 377 <= 31.0)
     Predict: 1.0
    Else (feature 377 > 31.0)
     If (feature 383 <= 28.0)
      Predict: 0.0
     Else (feature 383 > 28.0)
      Predict: 1.0
  Tree 9 (weight 1.0):
    If (feature 596 <= 45.5)
     If (feature 406 <= 9.5)
      Predict: 1.0
     Else (feature 406 > 9.5)
      Predict: 0.0
    Else (feature 596 > 45.5)
     If (feature 243 <= 1.0)
      Predict: 0.0
     Else (feature 243 > 1.0)
      Predict: 1.0
```


## Practice 5. Multilayer Peceptron Classifier.
##### Practice to run and document our observations of the example of the Spark documentation for Multilayer Peceptron Classifier.

Import libraries for MultilayerPerceptronClassifier and MulticlassClassificationEvaluator
```sh
    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
```
Create the DataFrame using the text file named "sample_multiclass_classification_data"
```sh
    val data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")
```
**Print result**
```sh
    scala>     val data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")
    22/11/26 19:55:38 WARN LibSVMFileFormat: 'numFeatures' option not specified, determining the number of features by going though the input. If you know the number in advance, please specify it via 'numFeatures' option to avoid the extra scan.
    data: org.apache.spark.sql.DataFrame = [label: double, features: vector]
```

Use randomSplit to create data of train and testing with of 60/40 dividers, create new variable called "train" and take the 0 position for the splits, finally, create a new "test" variable and asign the 1 position value of the splits. 
```sh
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
```
**Print Results**
```sh
    scala> val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    splits: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: double, features: vector], [label: double, features: vector])

    scala> val train = splits(0)
    train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

    scala> val test = splits(1)
    test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
```

Create a new layer array and define the neuronal network for the inputs 4, 5, 4, 3 values. 
```sh
    val layers = Array[Int](4, 5, 4, 3)
```
**Print Result**
```sh
    scala> val layers = Array[Int](4, 5, 4, 3)
    layers: Array[Int] = Array(4, 5, 4, 3)
```

Create new training variable and called "trainer" to assign parameters to set blocks for 128, seed for 1234L and max inter to 100. 
```sh
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100) 
```
**Print Result**
```sh
    scala> val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
    trainer: org.apache.spark.ml.classification MultilayerPerceptronClassifier = mlpc_a0d102e19090
```

Fit the trainer for the train conjunction.
```sh
    val model = trainer.fit(train)
```
**Print Result**
```sh
    scala> val model = trainer.fit(train)
    [Stage 43:>                                                         (0 + 1) / 1]
```
Get the result using the conjunction for transformation test
```sh
    val result = model.transform(test)
```
**Print Result**
```sh
    ERROR
```

Create predictionAndLabels from the result variable and create the label. 
```sh
    val predictionAndLabels = result.select("prediction", "label")
```
**Print Result**
```sh
    ERROR
```

Specify evaluator and create new instance of MulticlassClassificationEvaluator and set metric named "accuracy"
```sh
    val evaluator = new MulticlassClassificationEvaluator()
    .setMetricName("accuracy")
```
**Print Result**
```sh
    scala>   .setMetricName("accuracy")
    res21: evaluator.type = MulticlassClassificationEvaluator: uid=mcEval_920c519b33cf, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15
    
```
Print the accuracy and print the predictionAndLabels variable value.
```sh
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```
**Print Result**
```sh
    ERROR
```

## Evaluation Unit 2. 
##### Exercises to evaluate Unit 2 relative to Machine Learning Mllib of Spark and Google.

Import libraries needed for MachineLearning in Spark.
```sh
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
```

Load the file iris.cvs that we use to do the following operations and create the model for machine learning.
```sh
val iris = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("iris.csv")
```

**Print Result**
```sh
iris: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 3 more fields]
```

Define the value for the dataframe named df and transform the columns data types.
```sh
//val df = iris.withColumn("sepal_length",$"sepal_length".cast("double")).withColumn("sepal_width",$"sepal_width".cast("double")).withColumn("petal_length", $"petal_length".cast("double")).withColumn("petal_width", $"petal_width".cast("double"))
```

Use printSchema to print  the schema of the DataFrame  with column name and data type.
```sh
iris.printSchema()
```
**Print Result**
```sh
root
 |-- sepal_length: double (nullable = true)
 |-- sepal_width: double (nullable = true)
 |-- petal_length: double (nullable = true)
 |-- petal_width: double (nullable = true)
 |-- species: string (nullable = true)
 ```

Use the fucntion show() to display the contents of the DataFrame in a Table Row & Column Format. 
```sh
iris.show(5)
```

**Print Result**
```sh
+------------+-----------+------------+-----------+-------+
|sepal_length|sepal_width|petal_length|petal_width|species|
+------------+-----------+------------+-----------+-------+
|         5.1|        3.5|         1.4|        0.2| setosa|
|         4.9|        3.0|         1.4|        0.2| setosa|
|         4.7|        3.2|         1.3|        0.2| setosa|
|         4.6|        3.1|         1.5|        0.2| setosa|
|         5.0|        3.6|         1.4|        0.2| setosa|
+------------+-----------+------------+-----------+-------+
only showing top 5 rows
```

With describe function we can see the  mean, standard deviation, and minimum and maximum value for each numerical column 
```sh
iris.describe().show()
```

**Print Result**
```sh
+-------+------------------+-------------------+------------------+------------------+---------+
|summary|      sepal_length|        sepal_width|      petal_length|       petal_width|  species|
+-------+------------------+-------------------+------------------+------------------+---------+
|  count|               150|                150|               150|               150|      150|
|   mean| 5.843333333333335| 3.0540000000000007|3.7586666666666693|1.1986666666666672|     null|
| stddev|0.8280661279778637|0.43359431136217375| 1.764420419952262|0.7631607417008414|     null|
|    min|               4.3|                2.0|               1.0|               0.1|   setosa|
|    max|               7.9|                4.4|               6.9|               2.5|virginica|
+-------+------------------+-------------------+------------------+------------------+---------+
```

Next we merges multiple columns into a vector column named Features with VectorAssembler method.
```sh
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length", "petal_width")).setOutputCol("features")
```

Following assing to that parameter the data loaded from the iris file.
```sh
val features = assembler.transform(iris)
```

**Print Result**
```sh
features: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 4 more fields]
```

Converts string values to indexes for the categorical features
```sh
val indexerLabel = new StringIndexer().setInputCol("species").setOutputCol("indexedLabel").fit(features)
```

**Print Result**
```sh
indexerLabel: org.apache.spark.ml.feature.StringIndexerModel = StringIndexerModel: uid=strIdx_337483116212, handleInvalid=error
```

Create indexes for the vector of categorical features
```sh
val indexerFeatures = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4)
```

**Print Result**
```sh
indexerFeatures: org.apache.spark.ml.feature.VectorIndexer = vecIdx_5dc3c4038373
```

 Split the source data, using some of it to train the model and reserving some to test the trained model.
In this exercise, we will use 70% of the data for training, and reserve 30% for testing.

```sh
val Array(training, test) = features.randomSplit(Array(0.7, 0.3), seed = 12345)
```

**Print Result**
```sh
training: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [sepal_length: double, sepal_width: double ... 4 more fields]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [sepal_length: double, sepal_width: double ... 4 more fields]
```

Define the value layers that contains the array refering to the inputs layers (the inputs columns ) the hide layers and the output layer (number of species).
```sh
val layers = Array[Int](4, 5, 4, 3)
```

**Print Result**
```sh
layers: Array[Int] = Array(4, 5, 4, 3)
```

Used the Classifier trainer based on the Multilayer Perceptron. Number of inputs has to be equal to the size of feature vectors. Number of outputs has to be equal to the total number of labels.
```sh
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128).setSeed(1234).setMaxIter(100)
```

**Print Result**
```sh
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_fdf4b3531777
```

Create new IndexToString to set input column called "predictedLabel", and convert indexed labels back to original labels
```sh
val converterLabel = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(indexerLabel.labels)
```

**Print Result**
```sh
warning: one deprecation (since 3.0.0); for details, enable `:setting -deprecation' or `:replay -deprecation'
converterLabel: org.apache.spark.ml.feature.IndexToString = idxToStr_6a03c3861b36
```

Create new Pipeline with the following elements: indexerLabel, indexerFeatures, trainer, converterLabel.
A pipeline consists of a a series of transformer and estimator stages that typically prepare a DataFrame for modeling and then train a predictive model.
```sh
val pipeline = new Pipeline().setStages(Array(indexerLabel, indexerFeatures, trainer, converterLabel))
```
**Print Result**
```sh
pipeline: org.apache.spark.ml.Pipeline = pipeline_8227ee919b12
```

Use fit() function to the DataFrame and produces the Model, which is a Transformer. 
```sh
val model = pipeline.fit(training)
```
**Print Result**
```sh
model: org.apache.spark.ml.PipelineModel = pipeline_8227ee919b12
```

Converts the DataFrame with the test results
```sh
val results = model.transform(test)
```

**Print Result**
```sh
results: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 10 more fields]
```

Define the evaluator value whit MulticlassClassificationEvaluator method and define the parameters for the labels, prediction column and set the metric name.
```sh
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
```

**Print Result**
```sh
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_b852b9c85b9c, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15
```

Next evaluate the accuracy using  the results data into evaluator.
```sh
val accuracy = evaluator.evaluate(results)
```

**Print Result**
```sh
accuracy: Double = 0.9705882352941176
```

Finally print the accuracy and the error values.
```sh
println("Accuracy = " + accuracy)
```

**Print Result**
```sh
Accuracy = 0.9705882352941176
```

```sh
println("Error = " + (1.0 - accuracy))
```

**Print Result**
```sh
Error = 0.02941176470588236
```