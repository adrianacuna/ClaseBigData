import org.apache.spark.sql.SparkSession
package org.apache.spark.sql

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

//Chain of multiple dataframes operations
//Filter by Car_Model that begin with "Ma" and mpg is equal to 21
df
    .filter("Car_Model like 'Ma%'")
    .filter("mpg == 21 ")
    .show(10)


//We can create subsets of df data, using filter operartion to assign the values
// Using filter method slice our dataframe df with rows where the hp are equal o greater than 110 and less or equal than 180
val dfhpSubSet = df.filter("hp >= 110 and hp <= 180").toDF()
  dfhpSubSet.show()


//Spark support a number of join, in this example we us right outer join
//right outer join by joining df and dfhpSubSet
df
    .join(dfhpSubSet, Seq("cyl"), "right_outer")
    .show(10)

//With the dataframe df and the function avg, we calculate the average of the hp column
df
    .select(avg("hp"))
    .show()

//Whit the function max we can find car with the best mpg (miles per gallon)
df
    .select(max("mpg"))
    .show()

//For advanced statistics spark have stat functions, with freqItems method we can find 
//frequent items in the cyl column.
val dfFreCyl = df.stat.freqItems(Seq("cyl"))
  dfFreCyl.show()

// We can check if a column exist with the fucntion containts
//the column method can be used to return an array of type string
val dratColumnExists = df.columns.contains("drat")
  println(s"La columna drat existe = $dratColumnExists")

//Using distinct we can remove duplicate rows on dataframe 
val distinctDF = df.distinct()
println("Distinct count: "+distinctDF.count())
distinctDF.show(false)

//Alternatively we can also use dropDuplicates function wich create
// a new dataframe without duplicate rows
val df3 = df.dropDuplicates()
println("Distinct count: "+df2.count())
df2.show(false)