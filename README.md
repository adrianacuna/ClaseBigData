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