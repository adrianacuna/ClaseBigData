# ClaseBigData from unit1 branch
### Content menu
+ [Practice 1](#practice-1-git-basis)
+ [Practice 2](#practice-2-complete-git-flow-and-structure)
+ [Practice 3](#practice-3-scala-basis)
+ Practice 4
+ [Practice 5](#practice-5-code-analysis-for-scala-basics-in-session_6scala-teacher-file)
## Practice 1. Git basis. 
###### Practice to unclock the first level for the Introduction to GitCommits. 
[Learn Git Branching ](https://learngitbranching.js.org/)

**Result**
![Introduction Sequence](/unit1/assets/images/Practice1Image.png "Introduction Sequence")

## Practice 2. Complete Git flow and structure. 
###### View folders structure for the unit 1 into Git respository

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

###### 1. Create an algorithm in scala to calculate the **radius** for a circle.

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
###### 2. Create an algorithm in scala to determine if the number is **prime number**.

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
###### 3. For the static variable **bird = "tweet"**, use the string interpolation to print **"Estoy escribiendo un tweet"**.

We need to concatenate the bird variable string in the main message defined in the variable.
```sh
    val bird = "tweet"
    val message = s"Estoy ecribiendo un ${bird}"
```
**Print result**
```sh
    message: String = Estoy ecribiendo un tweet
```
###### 4. For the static variable **message**, use slice to extract the text **"Luke"**.

For the string variable we need to slice the text from character 5 to 9 and printed.
```sh
    val st = "Hola Luke yo soy tu padre!"
    st slice  (5,9)
```
**Print result**
```sh json
    res12: String = Luke
```
###### 5. What is the difference between **value** and **variable** in Scala?

This question is just open responses.

**Response**
```sh
    The value is inmutable and the variable can chage the value. 
```

## Practice 4.. 
## Practice 5. Code analysis for scala basics in Session_6.scala teacher file. 

###### 1. The following function named **isEven** receive one integer *number* to return a *boolean* result. 

The returning result is a validation if the number provided is divisible by two.

```sh
  def isEven(num:Int): Boolean = {
      return num%2 == 0
  }
  println(isEven(6))
  println(isEven(3))
```

**Print result**
*println(isEven(6)) -> true*
*println(isEven(3)) -> false*

###### 2. The following function analyze all elements of a list of integers, to print in screen if the numbers are even or odd.

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
*listEvens(l)*<br>
*1 is odd*<br>
*2 is even*<br>
*3 is odd*<br>
*4 is even*<br>
*5 is odd*<br>
*6 is even*<br>
*7 is odd*<br>
*8 is even*<br>
*String = Done*<br>

*listEvens(l2)*<br>
*4 is even*<br>
*3 is odd*<br>
*22 is even*<br>
*55 is odd*<br>
*7 is odd*<br>
*8 is even*<br>
*String = Done<br>

**Print result**<br>
*println(isEven(6)) -> true*<br>
*println(isEven(3)) -> false*<br>

###### 3. This function is for validate if the number of the list is equal to 7, if the result is **true** the function will add 14 with a sum, **if not** the result will sum the parameter number and the result saved in the variable.  

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
**Print result**<br>
*println(afortunado(af)) -> 29*<br>


###### 4. The next function called balance receives as a parameter a list of integers and returns a Boolean. It will be TRUE if the sum of the digits in the first half of it is equal to the sum of the digits in the second half. otherwise, it returns false.

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

**Print result**<br>
*balance(bl) -> Boolean = true*<br>
*balance(bl2) -> Boolean = true*<br>
*balance(bl3) -> Boolean = false*<br>

###### 5. This function is evaluated if the string parameter is a palindrome, if the string is able to read in reverse and the string is not deformed. 

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
**Print result**<br>
*println(palindromo(palabra)) -> true*<br>
*println(palindromo(palabra2)) -> true*<br>
*println(palindromo(palabra3)) -> false*<br>
