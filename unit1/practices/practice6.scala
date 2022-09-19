// Algorithm 1
def fib (n:Int): Int = {
    if ( n < 2 ){
        return n
    } else {
        return fib (n - 1) + fib (n - 2)
    }
}

// Algorithm 2
def fib2( n : Int ) : Int = {
  var a = 0
  var b = 1
  var i = 0	  
 
  while( i < n ) {
    val c = a + b
    a = b
    b = c
    i = i + 1
  } 
  return a
}


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