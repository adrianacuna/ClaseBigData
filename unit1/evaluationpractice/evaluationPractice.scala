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
//10. ¿Cuál es el máximo y mínimo de la columna “Volumen”?
//11. Con Sintaxis Scala/Spark $ conteste lo siguiente:
//a) ¿Cuántos días fue la columna “Close” inferior a $ 600?
//b) ¿Qué porcentaje del tiempo fue la columna “High” mayor que $ 500?
//c) ¿Cuál es la correlación de Pearson entre columna “High” y la columna “Volumen”?
//d) ¿Cuál es el máximo de la columna “High” por año?
//e) ¿Cuál es el promedio de la columna “Close” para cada mes del calendario?