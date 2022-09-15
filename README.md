# ClaseBigData from unit1 branch
## Practice 5. Code analysis for scala basics in Session_6.scala teacher file. 

The following function named **isEven** receive one integer *number* to return a *boolean* result. 

The returning result is a validation if the number provided is divisible by two.

``def isEven(num:Int): Boolean = {<br>
     return num%2 == 0<br>
}<br>
def isEven(num:Int): num%2 == 0<br>
println(isEven(6))<br>
println(isEven(3))<br>``

**Print result**<br>
*println(isEven(6)) -> true*<br>
*println(isEven(3)) -> false*<br>