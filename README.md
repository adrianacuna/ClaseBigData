# ClaseBigData from unit3 branch
### Content menu
+ [Evaluation Unit 3](#evaluation-unit-3)

## Evaluation Unit 3. 
##### Exercises for evaluatation Unit 3 relative to Machine Learning Multilayer Perceptron.

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

Create the columns variable to select the main columns of the main file imported, adding validation to transform the "Yes" and "No" strings to "1" and "0" respective. 
```sh
    val cols = data.select("age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y").withColumn("x", when(col("y") === "yes", 1).when(col("y") === "no", 0))
```
**Print Result**
```sh
    scala> val cols = data.select("age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y").withColumn("x", when(col("y") === "yes", 1).when(col("y") === "no", 0))
    cols: org.apache.spark.sql.DataFrame = [age: int, job: string ... 16 more fields]
```

See the schema for Cols variable.
```sh
    cols.printSchema()
```
**Print Result**
```sh
    scala> cols.printSchema()
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
    scala> cols.head(1)
    res2: Array[org.apache.spark.sql.Row] = Array([58,management,married,tertiary,no,2143,yes,no,unknown,5,may,261,1,-1,0,unknown,no,0])
```
Adding metadata for label column assigned to mlpdataall variable
```sh
    val mlpdataall = (cols.select(cols("x").as("label"), $"age", $"job",$"marital", $"education", $"day", $"month", $"duration", $"campaign", $"pdays", $"previous", $"poutcome"))
```
**Print Result**
```sh
    scala> val mlpdataall = (cols.select(cols("x").as("label"), $"age", $"job",$"marital", $"education", $"day", $"month", $"duration", $"campaign", $"pdays", $"previous", $"poutcome"))
    mlpdataall: org.apache.spark.sql.DataFrame = [label: int, age: int ... 10 more fields]
```
Clean data and remove NAs for mlpdata variable. 
```sh
    val mlpdata = mlpdataall.na.drop()
```
**Print Result**
```sh
    scala> val mlpdata = mlpdataall.na.drop()
    mlpdata: org.apache.spark.sql.DataFrame = [label: int, age: int ... 10 more fields]
```
Convert string columns values to numerical (integer) values. 
```sh
    val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
    val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex")
    val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
    val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex")
    val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutcomeIndex")
```
**Print Result**
```sh
    scala> val jobIndexer = new StringIndexer().setInputCol("job").setOutputCol("jobIndex")
    jobIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_de9cd00d6a32

    scala> val maritalIndexer = new StringIndexer().setInputCol("marital").setOutputCol("maritalIndex")
    maritalIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_a001f3fe0e68

    scala> val educationIndexer = new StringIndexer().setInputCol("education").setOutputCol("educationIndex")
    educationIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_6ab5c9f4313e

    scala> val monthIndexer = new StringIndexer().setInputCol("month").setOutputCol("monthIndex")
    monthIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_4a2968836b2c

    scala> val poutcomeIndexer = new StringIndexer().setInputCol("poutcome").setOutputCol("poutcomeIndex")
    poutcomeIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_693d21a5f9c5
```
Conver the numerical (integer) values to encoding binary 0-1
```sh
    val jobEncoder = new OneHotEncoder().setInputCol("jobIndex").setOutputCol("JobVec")
    val maritalEncoder = new OneHotEncoder().setInputCol("maritalIndex").setOutputCol("MaritalVec")
    val educationalEncoder = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("EducationVec")
    val monthEncoder = new OneHotEncoder().setInputCol("monthIndex").setOutputCol("monthVec")
    val poutcomeEncoder = new OneHotEncoder().setInputCol("poutcomeIndex").setOutputCol("PoutcomeVec")
    
```
**Print Result**
```sh
    scala> val jobEncoder = new OneHotEncoder().setInputCol("jobIndex").setOutputCol("JobVec")
    jobEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_166bb4ef53e2

    scala> val maritalEncoder = new OneHotEncoder().setInputCol("maritalIndex").setOutputCol("MaritalVec")
    maritalEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_5baf0a548a83

    scala> val educationalEncoder = new OneHotEncoder().setInputCol("educationIndex").setOutputCol("EducationVec")
    educationalEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_7f37cc5e81c3

    scala> val monthEncoder = new OneHotEncoder().setInputCol("monthIndex").setOutputCol("monthVec")
    monthEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_6f7381690102

    scala> val poutcomeEncoder = new OneHotEncoder().setInputCol("poutcomeIndex").setOutputCol("PoutcomeVec")
    poutcomeEncoder: org.apache.spark.ml.feature.OneHotEncoder = oneHotEncoder_73771dbc66c8
```
Merge multiple columns into a vector column named Features with VectorAssembler method.
```sh
    val assembler = new VectorAssembler().setInputCols(Array("age","JobVec","MaritalVec","EducationVec","day","monthVec","duration","campaign","pdays","previous","PoutcomeVec")).setOutputCol("features")
```
**Print Result**
```sh
    scala> val assembler = new VectorAssembler().setInputCols(Array("age","JobVec","MaritalVec","EducationVec","day","monthVec","duration","campaign","pdays","previous","PoutcomeVec")).setOutputCol("features")
    assembler: org.apache.spark.ml.feature.VectorAssembler = VectorAssembler: uid=vecAssembler_3df8e38ce1e1, handleInvalid=error, numInputCols=11
```

Split the data and define the percentage of each object, train and test. 
```sh
    val splits = mlpdata.randomSplit(Array(0.7, 0.3), seed = 1234L)

    val train = splits(0)
    val test = splits(1)
    
```
**Print Result**
```sh
    scala> val splits = mlpdata.randomSplit(Array(0.7, 0.3), seed = 1234L)
    splits: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: int, age: int ... 10 more fields], [label: int, age: int ... 10 more fields])

    scala> val train = splits(0)
    train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, age: int ... 10 more fields]

    scala> val test = splits(1)
    test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: int, age: int ... 10 more fields]
```
Define the layers for the neural network: input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 2 (classes)
```sh
    val layers = Array[Int](4, 5, 4, 2)
```
**Print Result**
```sh
    scala> val layers = Array[Int](4, 5, 4, 2)
    layers: Array[Int] = Array(4, 5, 4, 2)
```
Create the trainer and definee parameters: Layers, Feature, Block, Seed and Max Iter. 
```sh
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setFeaturesCol("features").setBlockSize(128).setSeed(1234L).setMaxIter(100)
```
**Print Result**
```sh
    scala> val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setFeaturesCol("features").setBlockSize(128).setSeed(1234L).setMaxIter(100)
    trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_62fb967cbcc2
```
Train the model variable using the trainer created before and Fit. 
```sh
    val model = trainer.fit(train)
```
**Print Result**
```sh
    ERROR = scala> val model = trainer.fit(train)
    java.lang.IllegalArgumentException: features does not exist. Available: label, age, job, marital, education, day, month, duration, campaign, pdays, previous, poutcome
```
Transfor the model using test variable, select the prediction and label and set the metrict name to accuracy. 
```sh
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
```
**Print Result**
```sh
    scala> val result = model.transform(test)
    <console>:32: error: not found: value model
        val result = model.transform(test)
                        ^
    scala> val predictionAndLabels = result.select("prediction", "label")
    <console>:31: error: not found: value result
        val predictionAndLabels = result.select("prediction", "label")
                                    ^
    scala> val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = MulticlassClassificationEvaluator: uid=mcEval_1754edb7971b, metricName=accuracy, metricLabel=0.0, beta=1.0, eps=1.0E-15
```

Define the print line and show the result of prediction and labels
```sh
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```
**Print Result**
```sh
    scala> println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
    <console>:33: error: not found: value predictionAndLabels
       println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```