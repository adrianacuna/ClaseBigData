# ClaseBigData from unit1 branch
## Practice 5. Code analysis for scala basics in Session_6.scala teacher file. 

###### 1. The following function named **isEven** receive one integer *number* to return a *boolean* result. 

The returning result is a validation if the number provided is divisible by two.

```
  def isEven(num:Int): Boolean = {
      return num%2 == 0
  }
  def isEven(num:Int): num%2 == 0
  println(isEven(6))
  println(isEven(3))
```

**Print result**
*println(isEven(6)) -> true*
*println(isEven(3)) -> false*

##The following function analyze all elements of a list of integers, to print in screen if the numbers are even or odd.

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
## Two different lists are created with different values, list 1 and list 12, then the function listEven is called

```sh
val l = List(1,2,3,4,5,6,7,8)
val l2 = List(4,3,22,55,7,8)
listEvens(l)
listEvens(l2)

```
**Print result**
*1 is odd*
*2 is even*
*3 is odd*
*4 is even*
*5 is odd*
*6 is even*
*7 is odd*
*8 is even*
*res0: String = Done*

*4 is even*
*3 is odd*
*22 is even*
*55 is odd*
*7 is odd*
*8 is even*
*res9: String = Done*
**Print result**<br>
*println(isEven(6)) -> true*<br>
*println(isEven(3)) -> false*<br>

###### 3. This function is for validate if the number of the list is equal to 7, if the result is **true** the function will add 14 with a sum, **if not** the result will sum the parameter number and the result saved in the variable.  
```
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
<<<<<<< HEAD
=======

###### 5. This function is evaluated if the string parameter is a palindrome, if the string is able to read in reverse and the string is not deformed. 

The reverse is the correct function to read the string parameter from the right to the left. 

```
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
>>>>>>> 3fb0f00749d7c39473afaf0617486c00453f73e4
