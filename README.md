# ClaseBigData from unit1 branch
### Content menu
+ [Practice 1](#practice-1-git-basis)
+ [Practice 2](#practice-2-complete-git-flow-and-structure)
+ [Practice 3](#practice-3-scala-basis)
+ [Practice 4](#practice-4-scala-collections)
+ [Practice 5](#practice-5-code-analysis-for-scala-basics-in-session_6scala-teacher-file)
+ [Practice 6](#practice-6-Implementation-of-Fibonacci-series-algorithms)
+ [Practice 7](#practice-7-Aggregate-function-for-spark-dataframes)
+ [Evaluation Unit 1](#evaluation-unit-1)

## Practice 1. Git basis. 
##### Practice to unclock the first level for the Introduction to GitCommits. 
[Learn Git Branching ](https://learngitbranching.js.org/)

**Result**
![Introduction Sequence](/unit1/assets/images/Practice1Image.png "Introduction Sequence")

## Practice 2. Complete Git flow and structure. 
##### View folders structure for the unit 1 into Git respository

- **Unit 1**
    - evaluationnpractice
        - evaluationPractice.scala
    - homeworks
        - homework1.scala
    - practices
        - practice1.scala
        - practice2.scala
        - practice3.scala
        - practice4.scala
        - practice5.scala
        - practice6.scala
        - practice7.scala
    - README.md

## Practice 3. Scala basis. 

##### 1. Create an algorithm in scala to calculate the **radius** for a circle.

We need to define the main circurference for the circle and defice the PI variable to creater the formula and calculate the radius. 
```sh
    val circ = 25
    val pi = 3.1416
    val rad = (circ/(2 * pi))
    val res = s"El radio del circulo es ${rad}"
```
**Print result**
```sh
    res: String = El radio del circulo es 3.97886427298192
```
##### 2. Create an algorithm in scala to determine if the number is **prime number**.

Creation for the main variables and define the number, create the IF conditions to validate if the result is prime number and print line. 
```sh
    val numero = 10
    val esPrimo = numero % 2
    if (esPrimo == 0) {
        val primo = s"El numero ${numero} es par"
        println(primo)
    } else {
        val primo = s"El numero ${numero} es primo"
        println(primo)
    }
```
**Print result**
```sh
    El numero 10 es par
```
##### 3. For the static variable **bird = "tweet"**, use the string interpolation to print **"Estoy escribiendo un tweet"**.

We need to concatenate the bird variable string in the main message defined in the variable.
```sh
    val bird = "tweet"
    val message = s"Estoy ecribiendo un ${bird}"
```
**Print result**
```sh
    message: String = Estoy ecribiendo un tweet
```
##### 4. For the static variable **message**, use slice to extract the text **"Luke"**.

For the string variable we need to slice the text from character 5 to 9 and printed.
```sh
    val st = "Hola Luke yo soy tu padre!"
    st slice  (5,9)
```
**Print result**
```sh json
    res12: String = Luke
```
##### 5. What is the difference between **value** and **variable** in Scala?

This question is just a open response.

**Response**
```sh
    The value is inmutable and the variable can chage the value. 
```

## Practice 4. Scala Collections. 
##### 1. Create a list called **"Lista"** with the elements **"rojo"**, **"blanco"** and **"negro"**.

Just create a variable and put into the List the main values.
```sh
    var lista = List("Rojo", "Blanco", "Negro")
    lista
```
**Print result**
```json
    lista: List[String] = List(Rojo, Blanco, Negro)
```
##### 2. Modified the previous list called **"Lista"** and add 5 elements **"verde"**, **"amarillo"**, **"naranja"**, and **"perla"**.

Insert the new elements in the list using ++= to insert the values.
```sh
    lista.++=(List("verde", "amarillo", "azul","naranja","perla"))
    println(lista)
```
**Print result**
```json
    List(Rojo, Blanco, Negro, verde, amarillo, azul, naranja, perla)
```
##### 3. Consult elements from previous list called **"Lista"** get the elements **"verde"**, **"amarillo"**, and **"azul"**.

For the main variable lista, consult the indexes values for the values specified.
```sh
    lista(3)
    lista(4)
    lista(5)
```
**Print result**
```
    scala> lista(3)
    res19: String = verde

    scala> lista(4)
    res20: String = amarillo

    scala> lista(5)
    res21: String = azul
```
##### 4. Create an **array number** ranged from **1 to 1000** with jumps by **5**.

We need to use the function range and specify the range numbers and the jumps.
```sh
    Array.range(1, 10000, 5)
```
**Print result**
```
    res22: Array[Int] = Array(1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 201, 206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266, 271, 276, 281, 286, 291, 296, 301, 306, 311, 316, 321, 326, 331, 336, 341, 346, 351, 356, 361, 366, 371, 376, 381, 386, 391, 396, 401, 406, 411, 416, 421, 426, 431, 436, 441, 446, 451, 456, 461, 466, 471, 476, 481, 486, 491, 496, 501, 506, 511, 516, 521, 526, 531, 536, 541, 546, 551, 556, 561, 566, 571, 576, 581, 586, 591, 596, 601, 606, 611, 616, 621, 626, 631, 636, 641, 646, 651, 656, 661, 666, 671, 676, 681, 686, 691, 696, 701, 706, 711, 716, 721, 726, 731, 736, 741, 746, 751, 756, 761, 76...
```
##### 5. Get the unique elements for the list Lista(1,3,3,4,6,7,3,7) using the set conversion.

Create a list variable and define the default values, we need to use the distinct method to remove duplications and set the list.
```sh
    val lista= List(1,3,3,4,6,7,3,7)
    val uniq = lista.distinct
    lista.toSet
```
**Print result**
```
    res23: scala.collection.immutable.Set[Int] = Set(1, 6, 7, 3, 4)
```
##### 6. Create a mutable map called **"nombres"** and add the values **"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"**.

Define the mutable list using the collection methods and add the main values with the ages.
```sh
    val nombres = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana",27))
```
**Print result**
```
    nombres: scala.collection.mutable.Map[String,Int] = Map(Susana -> 27, Ana -> 23, Luis -> 24, Jose -> 20)
```
##### 6a. In the previoud map called **"nombres"** print all **key values**.

Just print the key values using the method keys for a map collection.
```sh
    nombres.keys
```
**Print result**
```
   res24: Iterable[String] = Set(Susana, Ana, Luis, Jose)
```
##### 6b. In the previoud map called **"nombres"** add a new value **"Miguel", 23**.

Using the adding way of += to the main map to include the new values. 
```sh
    nombres += ("Miguel" -> 23)
    nombres.values
```
**Print result**
```
   res26: Iterable[Int] = HashMap(27, 23, 23, 24, 20)
```
## Practice 5. Code analysis for scala basics in Session_6.scala teacher file. 

##### 1. The following function named **isEven** receive one integer *number* to return a *boolean* result. 

The returning result is a validation if the number provided is divisible by two.

```sh
  def isEven(num:Int): Boolean = {
      return num%2 == 0
  }
  println(isEven(6))
  println(isEven(3))
```

**Print result**
```sh
    scala>   println(isEven(6))
    true

    scala>   println(isEven(3))
    false
```

##### 2. The following function analyze all elements of a list of integers, to print in screen if the numbers are even or odd.

Then enters into a for cycle to do the mod operation of each element of the given list, if the result of the operation is 0, prints "is even" otherwise prints "is odd". Finally returns the string "Done" when the cycle ends.   

```sh
    def listEvens(list:List[Int]): String ={
        for(n <- list){
            if(n%2==0){
                println(s"$n is even")
            }else{
                println(s"$n is odd")
            }
        }
        return "Done"
    }
```
Two different lists are created with different values, list 1 and list 12, then the function listEven is called

```sh
val l = List(1,2,3,4,5,6,7,8)
val l2 = List(4,3,22,55,7,8)

```
**Print result**
```sh
    scala> listEvens(l)
    1 is odd
    2 is even
    3 is odd
    4 is even
    5 is odd
    6 is even
    7 is odd
    8 is even
    res33: String = Done
```
```sh
    scala> listEvens(l2)
    4 is even
    3 is odd
    22 is even
    55 is odd
    7 is odd
    8 is even
    res34: String = Done
```
**Print result**
```sh
    scala> println(isEven(6))
    true

    scala> println(isEven(3))
    false
```
##### 3. This function is for validate if the number of the list is equal to 7, if the result is **true** the function will add 14 with a sum, **if not** the result will sum the parameter number and the result saved in the variable.  

```sh
    def afortunado(list:List[Int]): Int={
        var res=0
        for(n <- list){
            if(n==7){
                res = res + 14
            }else{
                res = res + n
            }
        }
        return res
    }

    val af= List(1,7,7)
    println(afortunado(af))
```
**Print result**
```sh
    scala>     println(afortunado(af))
    29
```
##### 4. The next function called balance receives as a parameter a list of integers and returns a Boolean. It will be TRUE if the sum of the digits in the first half of it is equal to the sum of the digits in the second half. otherwise, it returns false.

First two variables are declared, primera and segunda, segunda contain the sum of all elements of the list, then in the cycle for, The primera variable is assigned the sum of the elements one by one, and the second variable is subtracted element by element.If primera is equal segunda return true otherwise return false.

```sh
    def balance(list:List[Int]): Boolean={
        var primera = 0
        var segunda = 0

        segunda = list.sum

        for(i <- Range(0,list.length)){
            primera = primera + list(i)
            segunda = segunda - list(i)

            if(primera == segunda){
                return true
            }
        }
        return false 
    }
```
Three different lists are created with different values, list bl, bl2 and bl3 , then the function balance is called

```sh
    val bl = List(3,2,1)
    val bl2 = List(2,3,3,2)
    val bl3 = List(10,30,90)
```
**Print result**

```sh
    scala> balance(bl)
    res38: Boolean = true

    scala> balance(bl2)
    res39: Boolean = true

    scala> balance(bl3)
    res40: Boolean = false
```
##### 5. This function is evaluated if the string parameter is a palindrome, if the string is able to read in reverse and the string is not deformed. 

The reverse is the correct function to read the string parameter from the right to the left. 

```sh
    def palindromo(palabra:String):Boolean ={
        return (palabra == palabra.reverse)
    }

    val palabra = "OSO"
    val palabra2 = "ANNA"
    val palabra3 = "JUAN"

    println(palindromo(palabra))
    println(palindromo(palabra2))
    println(palindromo(palabra3))
```
**Print result**
```sh
    scala>   println(palindromo(palabra))
    true

    scala>   println(palindromo(palabra2))
    true

    scala>   println(palindromo(palabra3))
    false
```
## Practice 6. Implementation of Fibonacci series algorithms.

#### Algorithm 1: For the first algorithms we implemented the pseudo code using descending recursive Fibonacci method.

First defined a method with if else cycle, if variable n is lower than 2, return n, in other case do the operation fib (n-1) + fib(n-2) Recursively

```sh
// Algorithm 1
def fib (n:Int): Int = {
    if ( n < 2 ){
        return n
    } else {
        return fib (n - 1) + fib (n - 2)
    }
}
```

**Print result**
```sh
    scala> fib(11)
    res0: Int: 99
```

#### 2. Algorithm 3: For this algorith we implemented the pseudo code using the iterative version to calcule fibbonacci series.

In this case we defined three variables wit the following values, a=0, b=1 and i=0. Next defined a while cycle with condition (1 < n) using variable i as counter, inside the cycle created a new variable c = a + b, next assigned the value of b to a, and to b the value of c, finally do the increment i iteratively.


```sh
// Algorithm 3
def fib3( n : Int ) : Int = {
  var a = 0
  var b = 1
  var i = 0	  
 
  while( i < n ) {
    val c = a + b
    a = b
    b = c
    i = i + 1
  } 
  return b
}
```

**Print result**
```sh
    scala> fib3(15)
    res1: Int: 987
```

#### Algorithm 4: This algorithm calculate the Fibonacci series whit iterative method with two variables.

This is very similar to the past one, we defined 3 variables a=0, b=1 and i=0, next with a while cycle with the condition i< n, assign to b the sum of b+a and assign to the result of subtraction b-a, finally do the increment i iteratively


```sh
// Algorithm 4
def fib4( n : Int ) : Int = {
  var a = 0
  var b = 1
  var i = 0	  
 
  while( i < n ) {
    b = b + a
    a = b - a
    i = i + 1
  } 
  return a
}
```

**Print result**
```sh
    scala> fib4(55)
    res2: Int: 2144908973
```

#### Algorithm 5: In this case the algorithm calculate the Fibonacci series using a vector.

As same in algorithm 1, we use and if cycle to return n if the number is lower than 2, if not did the following assignment using a vector. variable a equal to a new vector with range (0, n) + 1, b = Vector (0) and c = Vector (1). finally with a for cycle define the variable vector whit the following assignment Vector (i - 1) + Vector (i - 2) and return vector.


```sh
// Algorithm 5
def fib5 (n:Int): Int = {
    if ( n < 2 ){
        return n
    } else {
        var a = Vector(Range(0, n) + 1)
        var b = Vector(0) 
        var c = Vector(1)
        var range = Range(2, n + 1)
        for ( i <- range ){
            var vector = Vector(i - 1) + Vector(i - 2)
            return vector
        }
    }
}
```

**Print result**
```sh
    scala> fib5(55)
    res3: Int: 1497
```


## Practice 7. Aggregate function for spark dataframes

In this practice we test some Data frame function with spark using a csv. file

##### Filter all cars with the specific HP between 100 and 120 and group by VS. Display all the filters and collect in array and count the values

```sh
val filterHP = df.filter($"hp" >= "100" && $"hp" <= "120")
.groupBy($"vs".desc)

filterHP.show()
filterHP.collect()
filterHP.count()
```
**Print result**
```sh
scala> filterHP.show()
+--------------+----+---+-----+---+----+-----+-----+---+---+----+----+
|     Car_Model| mpg|cyl| disp| hp|drat|   wt| qsec| vs| am|gear|carb|
+--------------+----+---+-----+---+----+-----+-----+---+---+----+----+
|     Mazda RX4|21.0|  6|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
| Mazda RX4 Wag|21.0|  6|160.0|110| 3.9|2.875|17.02|  0|  1|   4|   4|
|  Lotus Europa|30.4|  4| 95.1|113|3.77|1.513| 16.9|  1|  1|   5|   2|
|    Volvo 142E|21.4|  4|121.0|109|4.11| 2.78| 18.6|  1|  1|   4|   2|
+--------------+----+---+-----+---+----+-----+-----+---+---+----+----+

scala> filterHP.collect()
res4: Array[org.apache.spark.sql.Row] = Array([Mazda RX4,21.0,6,160.0,110,3.9,2.62,16.46,0,1,4,4], [Mazda RX4 Wag,21.0,6,160.0,110,3.9,2.875,17.02,0,1,4,4], [Hornet 4 Drive,21.4,6,258.0,110,3.08,3.215,19.44,1,0,3,1], [Valiant,18.1,6,225.0,105,2.76,3.46,20.22,1,0,3,1], [Lotus Europa,30.4,4,95.1,113,3.77,1.513,16.9,1,1,5,2], [Volvo 142E,21.4,4,121.0,109,4.11,2.78,18.6,1,1,4,2])

scala> filterHP.count()
res5: Long = 6

```


##### Sum for all disp values of disp column. 
##### Display 10 records for the sumDisp. 

```sh
val sumDisp = df.select(sumDistinct("disp"))
sumDisp.show(10)
```
**Print result**
```sh
+------------------+
|sum(DISTINCT disp)|
+------------------+
| 6143.900000000001|
+------------------+
```

##### Search value passing the specific value in arguments to the function and search the values matching to return the values in true.

```sh
def searchMPG(value: Double): Unit = {
    val searching = df.select($"mpg" === value)
    searching.show()
}
searchMPG(22.8)
```

**Print result**
```sh
+------------+
|(mpg = 22.8)|
+------------+
|       false|
|       false|
|        true|
|       false|
|       false|
|       false|
|       false|
|       false|
|        true|
|       false|
|       false|
|       false|
|       false|
|       false|
|       false|
+------------+
only showing top 20 rows
```

##### Select and rename colums as a easy select 

```sh
val df2 = df.select(col("Car_Model").as("Name Car"),
                    col("mpg").as("Miles per gallon"),
                    col("cyl").as("Cylindres"),
                    col("hp").as("HP Force"),
                    col("disp").as("Available"))
df2.show(5)
```

**Print result**
```sh
+-----------------+----------------+---------+--------+---------+
|         Name Car|Miles per gallon|Cylindres|HP Force|Available|
+-----------------+----------------+---------+--------+---------+
|        Mazda RX4|            21.0|        6|     110|    160.0|
|    Mazda RX4 Wag|            21.0|        6|     110|    160.0|
|       Datsun 710|            22.8|        4|      93|    108.0|
|   Hornet 4 Drive|            21.4|        6|     110|    258.0|
|Hornet Sportabout|            18.7|        8|     175|    360.0|
+-----------------+----------------+---------+--------+---------+
only showing top 5 rows
```

##### Chain of multiple dataframes operations
##### Filter by Car_Model that begin with "Ma" and mpg is equal to 21

```sh
df
    .filter("Car_Model like 'Ma%'")
    .filter("mpg == 21 ")
    .show(10)
```

**Print result**
```sh
+-------------+----+---+-----+---+----+-----+-----+---+---+----+----+
|    Car_Model| mpg|cyl| disp| hp|drat|   wt| qsec| vs| am|gear|carb|
+-------------+----+---+-----+---+----+-----+-----+---+---+----+----+
|    Mazda RX4|21.0|  6|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
|Mazda RX4 Wag|21.0|  6|160.0|110| 3.9|2.875|17.02|  0|  1|   4|   4|
+-------------+----+---+-----+---+----+-----+-----+---+---+----+----+
```

##### We can create subsets of df data, using filter operartion to assign the values using filter method slice our dataframe df with rows where the hp are equal o greater than 110 and less or equal than 180 val dfhpSubSet = df.filter("hp >= 110 and hp <= 180").toDF()

```sh
val dfhpSubSet = df.filter("hp >= 110 and hp <= 180").toDF()
dfhpSubSet.show()
```

**Print result**
```sh
+-----------------+----+---+-----+---+----+-----+-----+---+---+----+----+
|        Car_Model| mpg|cyl| disp| hp|drat|   wt| qsec| vs| am|gear|carb|
+-----------------+----+---+-----+---+----+-----+-----+---+---+----+----+
|        Mazda RX4|21.0|  6|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
|    Mazda RX4 Wag|21.0|  6|160.0|110| 3.9|2.875|17.02|  0|  1|   4|   4|
|   Hornet 4 Drive|21.4|  6|258.0|110|3.08|3.215|19.44|  1|  0|   3|   1|
|Hornet Sportabout|18.7|  8|360.0|175|3.15| 3.44|17.02|  0|  0|   3|   2|
|         Merc 280|19.2|  6|167.6|123|3.92| 3.44| 18.3|  1|  0|   4|   4|
|        Merc 280C|17.8|  6|167.6|123|3.92| 3.44| 18.9|  1|  0|   4|   4|
|       Merc 450SE|16.4|  8|275.8|180|3.07| 4.07| 17.4|  0|  0|   3|   3|
|       Merc 450SL|17.3|  8|275.8|180|3.07| 3.73| 17.6|  0|  0|   3|   3|
|      Merc 450SLC|15.2|  8|275.8|180|3.07| 3.78| 18.0|  0|  0|   3|   3|
| Dodge Challenger|15.5|  8|318.0|150|2.76| 3.52|16.87|  0|  0|   3|   2|
|      AMC Javelin|15.2|  8|304.0|150|3.15|3.435| 17.3|  0|  0|   3|   2|
| Pontiac Firebird|19.2|  8|400.0|175|3.08|3.845|17.05|  0|  0|   3|   2|
|     Lotus Europa|30.4|  4| 95.1|113|3.77|1.513| 16.9|  1|  1|   5|   2|
|     Ferrari Dino|19.7|  6|145.0|175|3.62| 2.77| 15.5|  0|  1|   5|   6|
+-----------------+----+---+-----+---+----+-----+-----+---+---+----+----+
```

##### Spark support a number of join, in this example we us right outer join right outer join by joining df and dfhpSubSet

```sh
df
    .join(dfhpSubSet, Seq("cyl"), "right_outer")
    .show(10)
```

**Print result**
```sh
+---+--------------+----+-----+---+----+-----+-----+---+---+----+----+-------------+----+-----+---+----+-----+-----+---+---+----+----+
|cyl|     Car_Model| mpg| disp| hp|drat|   wt| qsec| vs| am|gear|carb|    Car_Model| mpg| disp| hp|drat|   wt| qsec| vs| am|gear|carb|
+---+--------------+----+-----+---+----+-----+-----+---+---+----+----+-------------+----+-----+---+----+-----+-----+---+---+----+----+
|  6|  Ferrari Dino|19.7|145.0|175|3.62| 2.77| 15.5|  0|  1|   5|   6|    Mazda RX4|21.0|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
|  6|     Merc 280C|17.8|167.6|123|3.92| 3.44| 18.9|  1|  0|   4|   4|    Mazda RX4|21.0|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
|  6|      Merc 280|19.2|167.6|123|3.92| 3.44| 18.3|  1|  0|   4|   4|    Mazda RX4|21.0|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
|  6|       Valiant|18.1|225.0|105|2.76| 3.46|20.22|  1|  0|   3|   1|    Mazda RX4|21.0|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
|  6|Hornet 4 Drive|21.4|258.0|110|3.08|3.215|19.44|  1|  0|   3|   1|    Mazda RX4|21.0|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
|  6| Mazda RX4 Wag|21.0|160.0|110| 3.9|2.875|17.02|  0|  1|   4|   4|    Mazda RX4|21.0|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
|  6|     Mazda RX4|21.0|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|    Mazda RX4|21.0|160.0|110| 3.9| 2.62|16.46|  0|  1|   4|   4|
|  6|  Ferrari Dino|19.7|145.0|175|3.62| 2.77| 15.5|  0|  1|   5|   6|Mazda RX4 Wag|21.0|160.0|110| 3.9|2.875|17.02|  0|  1|   4|   4|
|  6|     Merc 280C|17.8|167.6|123|3.92| 3.44| 18.9|  1|  0|   4|   4|Mazda RX4 Wag|21.0|160.0|110| 3.9|2.875|17.02|  0|  1|   4|   4|
|  6|      Merc 280|19.2|167.6|123|3.92| 3.44| 18.3|  1|  0|   4|   4|Mazda RX4 Wag|21.0|160.0|110| 3.9|2.875|17.02|  0|  1|   4|   4|
+---+--------------+----+-----+---+----+-----+-----+---+---+----+----+-------------+----+-----+---+----+-----+-----+---+---+----+----+
only showing top 10 rows
```


##### With the dataframe df and the function avg, we calculate the average of the hp column

```sh
df
    .select(avg("hp"))
    .show()
```

**Print result**
```sh
+--------+
| avg(hp)|
+--------+
|146.6875|
+--------+
```

##### Whit the function max we can find car with the best mpg (miles per gallon)

```sh
df
    .select(max("mpg"))
    .show()
```

**Print result**
```sh
+--------+
|max(mpg)|
+--------+
|    33.9|
+--------+
```

##### For advanced statistics spark have stat functions, with freqItems method we can find frequent items in the cyl column.

```sh
val dfFreCyl = df.stat.freqItems(Seq("cyl"))
  dfFreCyl.show()
```

**Print result**
```sh
+-------------+
|cyl_freqItems|
+-------------+
|    [8, 4, 6]|
+-------------+
```

##### We can check if a column exist with the fucntion containts the column method can be used to return an array of type string

```sh
val dratColumnExists = df.columns.contains("drat")
  println(s"La columna drat existe = $dratColumnExists")
```
**Print result**
```sh
La columna drat existe = true
```

##### Using distinct we can remove duplicate rows on dataframe 

```sh
val distinctDF = df.distinct()
println("Distinct count: "+distinctDF.count())
distinctDF.show(false)
```

**Print result**
```sh
Distinct count: 32

scala> distinctDF.show(false)
+------------------+----+---+-----+---+----+-----+-----+---+---+----+----+
|Car_Model         |mpg |cyl|disp |hp |drat|wt   |qsec |vs |am |gear|carb|
+------------------+----+---+-----+---+----+-----+-----+---+---+----+----+
|Fiat 128          |32.4|4  |78.7 |66 |4.08|2.2  |19.47|1  |1  |4   |1   |
|Maserati Bora     |15.0|8  |301.0|335|3.54|3.57 |14.6 |0  |1  |5   |8   |
|Merc 450SL        |17.3|8  |275.8|180|3.07|3.73 |17.6 |0  |0  |3   |3   |
|Pontiac Firebird  |19.2|8  |400.0|175|3.08|3.845|17.05|0  |0  |3   |2   |
|Dodge Challenger  |15.5|8  |318.0|150|2.76|3.52 |16.87|0  |0  |3   |2   |
|Ford Pantera L    |15.8|8  |351.0|264|4.22|3.17 |14.5 |0  |1  |5   |4   |
|Porsche 914-2     |26.0|4  |120.3|91 |4.43|2.14 |16.7 |0  |1  |5   |2   |
|Honda Civic       |30.4|4  |75.7 |52 |4.93|1.615|18.52|1  |1  |4   |2   |
|Toyota Corona     |21.5|4  |120.1|97 |3.7 |2.465|20.01|1  |0  |3   |1   |
|Valiant           |18.1|6  |225.0|105|2.76|3.46 |20.22|1  |0  |3   |1   |
|Merc 450SE        |16.4|8  |275.8|180|3.07|4.07 |17.4 |0  |0  |3   |3   |
|Duster 360        |14.3|8  |360.0|245|3.21|3.57 |15.84|0  |0  |3   |4   |
|Ferrari Dino      |19.7|6  |145.0|175|3.62|2.77 |15.5 |0  |1  |5   |6   |
|Mazda RX4         |21.0|6  |160.0|110|3.9 |2.62 |16.46|0  |1  |4   |4   |
|Hornet Sportabout |18.7|8  |360.0|175|3.15|3.44 |17.02|0  |0  |3   |2   |
|Cadillac Fleetwood|10.4|8  |472.0|205|2.93|5.25 |17.98|0  |0  |3   |4   |
|Lotus Europa      |30.4|4  |95.1 |113|3.77|1.513|16.9 |1  |1  |5   |2   |
|Volvo 142E        |21.4|4  |121.0|109|4.11|2.78 |18.6 |1  |1  |4   |2   |
|AMC Javelin       |15.2|8  |304.0|150|3.15|3.435|17.3 |0  |0  |3   |2   |
|Merc 240D         |24.4|4  |146.7|62 |3.69|3.19 |20.0 |1  |0  |4   |2   |
+------------------+----+---+-----+---+----+-----+-----+---+---+----+----+
only showing top 20 rows
```

##### Alternatively we can also use dropDuplicates function wich create a new dataframe without duplicate rows

```sh
val df3 = df.dropDuplicates()
println("Distinct count: "+df2.count())
df2.show(false)
```

**Print result**
```sh
Distinct count: 32

scala> df2.show(false)
+-------------------+----------------+---------+--------+---------+
|Name Car           |Miles per gallon|Cylindres|HP Force|Available|
+-------------------+----------------+---------+--------+---------+
|Mazda RX4          |21.0            |6        |110     |160.0    |
|Mazda RX4 Wag      |21.0            |6        |110     |160.0    |
|Datsun 710         |22.8            |4        |93      |108.0    |
|Hornet 4 Drive     |21.4            |6        |110     |258.0    |
|Hornet Sportabout  |18.7            |8        |175     |360.0    |
|Valiant            |18.1            |6        |105     |225.0    |
|Duster 360         |14.3            |8        |245     |360.0    |
|Merc 240D          |24.4            |4        |62      |146.7    |
|Merc 230           |22.8            |4        |95      |140.8    |
|Merc 280           |19.2            |6        |123     |167.6    |
|Merc 280C          |17.8            |6        |123     |167.6    |
|Merc 450SE         |16.4            |8        |180     |275.8    |
|Merc 450SL         |17.3            |8        |180     |275.8    |
|Merc 450SLC        |15.2            |8        |180     |275.8    |
|Cadillac Fleetwood |10.4            |8        |205     |472.0    |
|Lincoln Continental|10.4            |8        |215     |460.0    |
|Chrysler Imperial  |14.7            |8        |230     |440.0    |
|Fiat 128           |32.4            |4        |66      |78.7     |
|Honda Civic        |30.4            |4        |52      |75.7     |
|Toyota Corolla     |33.9            |4        |65      |71.1     |
+-------------------+----------------+---------+--------+---------+
only showing top 20 rows
```


## Evaluation unit 1.  

## Evaluation Practice
#### 1. Start a Spark session

```sh
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()
```
**Print result**
```sh
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@56a640a2
```

#### 2. Load the file Netflix Stock CSV to a dataframe named df.

```sh
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv")

import spark.implicits._
```

**Print result**
```sh
df: org.apache.spark.sql.DataFrame = [Date: timestamp, Open: double ... 5 more fields]
```

#### 3. What are the column names?

```sh
df.columns
```

**Print result**
```sh
res0: Array[String] = Array(Date, Open, High, Low, Close, Volume, Adj Close)
```

#### 4. How is the scheme?

```sh
df.printSchema()
```

**Print result**
```sh
root
 |-- Date: timestamp (nullable = true)
 |-- Open: double (nullable = true)
 |-- High: double (nullable = true)
 |-- Low: double (nullable = true)
 |-- Close: double (nullable = true)
 |-- Volume: integer (nullable = true)
 |-- Adj Close: double (nullable = true)
```

#### 5. Print the first 5 rows.

```sh
df.show(5)
```

**Print result**
```sh
+-------------------+----------+------------------+----------+-----------------+---------+------------------+
|               Date|      Open|              High|       Low|            Close|   Volume|         Adj Close|
+-------------------+----------+------------------+----------+-----------------+---------+------------------+
|2011-10-24 00:00:00|119.100002|120.28000300000001|115.100004|       118.839996|120460200|         16.977142|
|2011-10-25 00:00:00| 74.899999|         79.390001| 74.249997|        77.370002|315541800|11.052857000000001|
|2011-10-26 00:00:00|     78.73|         81.420001| 75.399997|        79.400002|148733900|         11.342857|
|2011-10-27 00:00:00| 82.179998| 82.71999699999999| 79.249998|80.86000200000001| 71190000|11.551428999999999|
|2011-10-28 00:00:00| 80.280002|         84.660002| 79.599999|84.14000300000001| 57769600|             12.02|
+-------------------+----------+------------------+----------+-----------------+---------+------------------+
only showing top 5 rows
```

#### 6. Use describe () method to learn about the dataframe.

```sh
df.describe().show()
```

**Print result**
```sh
+-------+------------------+------------------+------------------+------------------+--------------------+------------------+
|summary|              Open|              High|               Low|             Close|              Volume|         Adj Close|
+-------+------------------+------------------+------------------+------------------+--------------------+------------------+
|  count|              1259|              1259|              1259|              1259|                1259|              1259|
|   mean|230.39351086656092|233.97320872915006|226.80127876251044|  230.522453845909|2.5634836060365368E7|55.610540036536875|
| stddev|164.37456353264244| 165.9705082667129| 162.6506358235739|164.40918905512854| 2.306312683388607E7|35.186669331525486|
|    min|         53.990001|         55.480001|             52.81|              53.8|             3531300|          7.685714|
|    max|        708.900017|        716.159996|        697.569984|        707.610001|           315541800|        130.929993|
+-------+------------------+------------------+------------------+------------------+--------------------+------------------+
```

#### 7. Create a new dataframe with a new column called "HV Ratio" wich is the relationship between price from column "High" with the column "Volume" of stocks traded in one day.

```sh
val df2 = df.withColumn("HV Ratio", df("High")*df("Volume"))
```

**Print result**
```sh
scala> df2.show(5df2.show())
+-------------------+-----------------+------------------+----------+-----------------+---------+------------------+--------------------+
|               Date|             Open|              High|       Low|            Close|   Volume|         Adj Close|            HV Ratio|
+-------------------+-----------------+------------------+----------+-----------------+---------+------------------+--------------------+
|2011-10-24 00:00:00|       119.100002|120.28000300000001|115.100004|       118.839996|120460200|         16.977142|1.448895321738060...|
|2011-10-25 00:00:00|        74.899999|         79.390001| 74.249997|        77.370002|315541800|11.052857000000001|2.505086381754179...|
|2011-10-26 00:00:00|            78.73|         81.420001| 75.399997|        79.400002|148733900|         11.342857| 1.21099142867339E10|
|2011-10-27 00:00:00|        82.179998| 82.71999699999999| 79.249998|80.86000200000001| 71190000|11.551428999999999| 5.888836586429999E9|
|2011-10-28 00:00:00|        80.280002|         84.660002| 79.599999|84.14000300000001| 57769600|             12.02| 4.890774451539201E9|
+-------------------+-----------------+------------------+----------+-----------------+---------+------------------+--------------------+
only showing top 5 rows
```

#### 8.  What day had the higher peak for column "Open"?

```sh
df.orderBy($"Open".desc).show(1)
```

**Print result**
```sh
+-------------------+----------+----------+----------+----------+--------+----------+
|               Date|      Open|      High|       Low|     Close|  Volume| Adj Close|
+-------------------+----------+----------+----------+----------+--------+----------+
|2015-07-14 00:00:00|708.900017|711.449982|697.569984|702.600006|19736500|100.371429|
+-------------------+----------+----------+----------+----------+--------+----------+
only showing top 1 row
```

#### 9. What is the meaning of the "Close" clumn, in the context of financial information?
Respuesta: Close hace referencia al precio de una acción individual cuando la bolsa de valores cierra en un día en especifico

#### 10. What is the max and min of "Volume" column?

```sh
df.groupBy("Volume").max().show(1)
df.groupBy("Volume").min().show(1)
```

**Print result**
```sh
scala> df.groupBy("Volume").max().show(1)
+--------+---------+---------+---------+----------+-----------+-----------------+
|  Volume|max(Open)|max(High)| max(Low)|max(Close)|max(Volume)|   max(Adj Close)|
+--------+---------+---------+---------+----------+-----------+-----------------+
|59170300|67.059999|68.199999|65.120002| 66.560001|   59170300|9.508572000000001|
+--------+---------+---------+---------+----------+-----------+-----------------+
only showing top 1 row


scala> df.groupBy("Volume").min().show(1)
+--------+---------+---------+---------+----------+-----------+-----------------+
|  Volume|min(Open)|min(High)| min(Low)|min(Close)|min(Volume)|   min(Adj Close)|
+--------+---------+---------+---------+----------+-----------+-----------------+
|59170300|67.059999|68.199999|65.120002| 66.560001|   59170300|9.508572000000001|
+--------+---------+---------+---------+----------+-----------+-----------------+
only showing top 1 row
```


#### 11. With Syntax Scala/Spark anwser the following:
##### a) How many day the column "Close" was lower tan $600?
        
```sh
df.filter($"Close" < 600).count()
```
**Print result**
```sh
res12: Long = 1218
```

###### b) What is the percentage of the time that the column "High" was greater tan $500?
            
```sh
df.filter($"High" > 500).count()* 1.0/ df.count()*100
```
**Print result**
```sh
res13: Double = 4.924543288324067
```

##### c) What is the pearson corralation between "Hight" and "Volume" columns?
        
```sh
df.select(corr("High","Volume")).show()
```

**Print result**
```sh
+--------------------+
|  corr(High, Volume)|
+--------------------+
|-0.20960233287942157|
+--------------------+
```


##### d) What is the max value per year for the column "High"?
        
```sh
val df2 = df.withColumn("Year", year(df("Date")))
val dfamax = df2.select($"Year",$"High").groupBy("Year").max()
val max = dfamax.select($"Year",$"max(High)")
max.orderBy("Year").show()
```

**Print result**
```sh
+----+------------------+
|Year|         max(High)|
+----+------------------+
|2011|120.28000300000001|
|2012|        133.429996|
|2013|        389.159988|
|2014|        489.290024|
|2015|        716.159996|
|2016|129.28999299999998|
+----+------------------+
```

##### e) What is the average per month for the "Close" column?
        
```sh
val df3 = df.withColumn("Month", month(df("Date")))
val dfavgs = df3.groupBy("Month").mean()
dfavgs.select($"Month", $"avg(Close)").show()
```

**Print result**
```sh
+-----+------------------+
|Month|        avg(Close)|
+-----+------------------+
|   12| 199.3700942358491|
|    1|212.22613874257422|
|    6| 295.1597153490566|
|    3| 249.5825228971963|
|    5|264.37037614150944|
|    9|206.09598121568627|
|    4|246.97514271428562|
|    8|195.25599892727263|
|    7|243.64747528037387|
|   10|205.93297300900903|
|   11| 194.3172275445545|
|    2| 254.1954634020619|
+-----+------------------+
```