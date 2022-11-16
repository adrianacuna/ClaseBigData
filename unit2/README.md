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

Que el objeto assembler convierta los valores de entrada a un vector
Utilice el objeto VectorAssembler para convertir la columnas de entradas del df a una sola columna de salida de un arreglo llamado  "features"
Configure las columnas de entrada de donde se supone que leemos los valores.
Llamar a esto nuevo assambler.

```sh
val assembler = new VectorAssembler().setInputCols(Array("Avg Session Length","Time on App","Time on Website","Length of Membership")).setOutputCol("features")
```


Utilice el assembler para transform nuestro DataFrame a dos columnas: label and features

```sh
val output = assembler.transform(df).select($"label",$"features")
```

Crear un objeto para modelo de regresion linea.

```sh
val lr = new LinearRegression()
```

Ajuste el modelo para los datos y llame a este modelo lrModelo

```sh
val lrModelo = lr.fit(output)
```

Imprima the  coefficients y intercept para la regresion lineal

```sh
println(s"Coefficients: ${lrModelo.coefficients} Intercept: ${lrModelo.intercept}")
```

Resuma el modelo sobre el conjunto de entrenamiento imprima la salida de algunas metricas!
Utilize metodo .summary de nuestro  modelo para crear un objeto
llamado trainingSummary

```sh
val trainingSummary = lrModelo.summary
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
    
```

## Practice 2. Logistic Regression.
##### Practice to analyze and document the PracticalLogisticRegression.scala file.


## Practice 3. Decision Tree. 
##### Practice to run and document our observations of the example of the Spark documentation for Decision Tree Classifier.

##### 1. Create an algorithm in scala to calculate the **radius** for a circle.

## Practice 4. Random Forest.
##### Practice to run and document our observations of the example of the Spark documentation for Random Forest.

## Practice 4. Multilayer Peceptron Classifier.
##### Practice to run and document our observations of the example of the Spark documentation for Multilayer Peceptron Classifier.
