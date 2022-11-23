# ClaseBigData from unit2 branch
### Content menu
+ [Practice 1](#practice-1-linear-regression)
+ [Practice 2](#practice-2-logistic-regression)
+ [Practice 3](#practice-3-decision-tree)
+ [Practice 4](#practice-4-random-forest)
+ [Practice 5](#practice-5-multilayer-peceptron-classifier)
+ [Evaluation Unit 1](#evaluation-unit-2)

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

Inicie una simple Sesion Spark
```sh
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
```

Utilice Spark para el archivo csv Clean-Ecommerce.
```sh
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Clean-Ecommerce.csv")
```

Imprima el schema en el DataFrame.
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

Transforme el data frame para que tome la forma de ("label","features")
Importe VectorAssembler y Vectors

```sh
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
```

Renombre la columna Yearly Amount Spent como "label", tambien de los datos tome solo la columa numerica. Deje todo esto como un nuevo DataFrame que se llame df

```sh
val df = data.select(data("Yearly Amount Spent").as("label"),
$"Avg Session Length",$"Time on App",$"Time on Website",$"Length of Membership")
```
**Print result**
```sh
df: org.apache.spark.sql.DataFrame = [label: double, Avg Session Length: double ... 3 more fields]
```

Que el objeto assembler convierta los valores de entrada a un vector
Utilice el objeto VectorAssembler para convertir la columnas de entradas del df a una sola columna de salida de un arreglo llamado  "features"
Configure las columnas de entrada de donde se supone que leemos los valores.
Llamar a esto nuevo assambler.

```sh
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")
```
**Print result**
```sh
assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_b3c45a0cf4e8, handleInvalid=error, numInputCols=4
```

Utilice el assembler para transform nuestro DataFrame a dos columnas: label and features

```sh
val output = assembler.transform(df).select($"label",$"features")
```
**Print result**
```sh
output: org.apache.spark.sql.DataFrame = [label: double, features: vector]
```

Crear un objeto para modelo de regresion linea.

```sh
val lr = new LinearRegression()
```
**Print result**
```sh
lr: org.apache.spark.ml.regression.LinearRegression = linReg_022cc6b7f4ca
```

Ajuste el modelo para los datos y llame a este modelo lrModelo

```sh
val lrModelo = lr.fit(output)
```
**Print result**
```sh
lrModelo: org.apache.spark.ml.regression.LinearRegressionModel = LinearRegressionModel: uid=linReg_022cc6b7f4ca, numFeatures=4
```

Imprima the  coefficients y intercept para la regresion lineal

```sh
println(s"Coefficients: ${lrModelo.coefficients} Intercept: ${lrModelo.intercept}")
```
**Print result**
```sh
Coefficients: [25.734271084670716,38.709153810828816,0.43673883558514964,61.57732375487594] Intercept: -1051.5942552990748
```

Resuma el modelo sobre el conjunto de entrenamiento imprima la salida de algunas metricas!
Utilize metodo .summary de nuestro  modelo para crear un objeto
llamado trainingSummary

```sh
val trainingSummary = lrModelo.summary
```

**Print result**
```sh
trainingSummary: org.apache.spark.ml.regression.LinearRegressionTrainingSummary = org.apache.spark.ml.regression.LinearRegressionTrainingSummary@7d026083
```

Muestre los valores de residuals, el RMSE, el MSE, y tambien el R^2 .

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

##### 1. Create an algorithm in scala to calculate the **radius** for a circle.

## Practice 4. Random Forest.
##### Practice to run and document our observations of the example of the Spark documentation for Random Forest.

## Practice 4. Multilayer Peceptron Classifier.
##### Practice to run and document our observations of the example of the Spark documentation for Multilayer Peceptron Classifier.
