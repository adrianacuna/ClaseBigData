import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("CarData.csv")

import spark.implicits._

// Filter all cars with the specific HP between 100 and 120 and group by VS
// Display all the filters and collect in array and count the values
val filterHP = df.filter($"hp" >= "100" && $"hp" <= "120")
.groupBy($"vs".desc)

filterHP.show()
filterHP.collect()
filterHP.count()

// Sum for all disp values of disp column. 
// Display 10 records for the sumDisp. 
val sumDisp = df.select(sumDistinct("disp"))
sumDisp.show(10)

// Search value passing the specific value in arguments to the function
// and search the values matching to return the values in true.
def searchMPG(value: Double): Unit = {
    val searching = df.select($"mpg" === value)
    searching.show()
}
searchMPG(22.8)

// Select and rename colums as a easy select 
val df2 = df.select(col("Car_Model").as("Name Car"),
                    col("mpg").as("Miles per gallon"),
                    col("cyl").as("Cylindres"),
                    col("hp").as("HP Force"),
                    col("disp").as("Available"))
df2.show(5)

//


