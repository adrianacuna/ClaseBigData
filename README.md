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