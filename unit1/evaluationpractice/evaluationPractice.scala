//Evaluation Practice
//1. Comienza una simple sesión Spark.
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()
//2. Cargue el archivo Netflix Stock CSV en dataframe llamado df, haga que Spark infiera los tipos de datos.
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

import spark.implicits._

//3. ¿Cuáles son los nombres de las columnas?
df.columns

//4. ¿Cómo es el esquema?
df.printSchema()

//5. Imprime las primeras 5 renglones.
df.show(5)

//6. Usa el método describe () para aprender sobre el DataFrame.
df.describe().show()


//7. Crea un nuevo dataframe con una columna nueva llamada “HV Ratio” que es la relación que existe entre el precio de la columna “High” frente a la columna “Volumen” de acciones negociadas por un día. Hint - es una operación
val df2 = df.withColumn("HV Ratio", df("High")*df("Volume"))

//8. ¿Qué día tuvo el pico más alto en la columna “Open”?
df.orderBy($"Open".desc).show(1)

//9. ¿Cuál es el significado de la columna Cerrar “Close” en el contexto de información financiera, explíquelo no hay que codificar nada?
//Respuesta: Close hace referencia al precio de una acción individual cuando la bolsa de valores cierra en un día en especifico

//10. ¿Cuál es el máximo y mínimo de la columna “Volumen”?
df.groupBy("Volume").max().show(1)
df.groupBy("Volume").min().show(1)

//11. Con Sintaxis Scala/Spark $ conteste lo siguiente:
//a) ¿Cuántos días fue la columna “Close” inferior a $ 600?
df.filter($"Close" < 600).count()

//b) ¿Qué porcentaje del tiempo fue la columna “High” mayor que $ 500?
df.filter($"High" > 500).count()* 1.0/ df.count()*100

//c) ¿Cuál es la correlación de Pearson entre columna “High” y la columna “Volumen”?
df.select(corr("High","Volume")).show()

//d) ¿Cuál es el máximo de la columna “High” por año?
val df2 = df.withColumn("Year", year(df("Date")))
val dfamax = df2.select($"Year",$"High").groupBy("Year").max()
val max = dfamax.select($"Year",$"max(High)")
max.orderBy("Year").show()

//e) ¿Cuál es el promedio de la columna “Close” para cada mes del calendario?
val df3 = df.withColumn("Month", month(df("Date")))
val dfavgs = df3.groupBy("Month").mean()
dfavgs.select($"Month", $"avg(Close)").show()