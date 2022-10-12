# ClaseBigData from unit1 branch
### Content menu
+ [Practice 1](#practice-1-git-basis)
+ [Practice 2](#practice-2-complete-git-flow-and-structure)
+ [Practice 3](#practice-3-scala-basis)
+ [Practice 4](#practice-4-scala-collections)
+ [Practice 5](#practice-5-code-analysis-for-scala-basics-in-session_6scala-teacher-file)
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
## Practice 6. Implementation of Fibonacci series algorithms according to pseudo code from Wikipedia link

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

##### Sum for all disp values of disp column. 
##### Display 10 records for the sumDisp. 

```sh
val sumDisp = df.select(sumDistinct("disp"))
sumDisp.show(10)
```

##### Search value passing the specific value in arguments to the function and search the values matching to return the values in true.

```sh
def searchMPG(value: Double): Unit = {
    val searching = df.select($"mpg" === value)
    searching.show()
}
searchMPG(22.8)
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

##### Chain of multiple dataframes operations
##### Filter by Car_Model that begin with "Ma" and mpg is equal to 21

```sh
df
    .filter("Car_Model like 'Ma%'")
    .filter("mpg == 21 ")
    .show(10)
```


##### We can create subsets of df data, using filter operartion to assign the values using filter method slice our dataframe df with rows where the hp are equal o greater than 110 and less or equal than 180 val dfhpSubSet = df.filter("hp >= 110 and hp <= 180").toDF()

```sh
dfhpSubSet.show()
```


##### Spark support a number of join, in this example we us right outer join right outer join by joining df and dfhpSubSet

```sh
df
    .join(dfhpSubSet, Seq("cyl"), "right_outer")
    .show(10)
```
##### With the dataframe df and the function avg, we calculate the average of the hp column

```sh
df
    .select(avg("hp"))
    .show()
```

##### Whit the function max we can find car with the best mpg (miles per gallon)

```sh
df
    .select(max("mpg"))
    .show()
```
##### For advanced statistics spark have stat functions, with freqItems method we can find frequent items in the cyl column.

```sh
val dfFreCyl = df.stat.freqItems(Seq("cyl"))
  dfFreCyl.show()
```
##### We can check if a column exist with the fucntion containts the column method can be used to return an array of type string

```sh
val dratColumnExists = df.columns.contains("drat")
  println(s"La columna drat existe = $dratColumnExists")
```

##### Using distinct we can remove duplicate rows on dataframe 

```sh
val distinctDF = df.distinct()
println("Distinct count: "+distinctDF.count())
distinctDF.show(false)
```
##### Alternatively we can also use dropDuplicates function wich create a new dataframe without duplicate rows

```sh
val df3 = df.dropDuplicates()
println("Distinct count: "+df2.count())
df2.show(false)
```


## Evaluation unit 1.  