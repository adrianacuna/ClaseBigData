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