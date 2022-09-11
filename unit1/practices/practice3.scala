//Practice 3
//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo
val circ = 25
val pi = 3.1416
val rad = (circ/(2 * pi))
val res = s"El radio del circulo es ${rad}"
//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
val numero = 10
val esPrimo = numero % 2
if (esPrimo == 0) {
  val primo = s"El numero ${numero} es par"
  println(primo)
} else {
  val primo = s"El numero ${numero} es primo"
  println(primo)
}

//3. Dada la variable bird = "tweet", utiliza interpolacion de string para
//   imprimir "Estoy ecribiendo un tweet"
val bird = "tweet"
val message = s"Estoy ecribiendo un ${bird}"


//4. Dada la variable mensaje = value utiliza slilce para extraer la
//   secuencia "Luke"
val st = "Hola Luke yo soy tu padre!"
st slice  (5,9)

//5. Cual es la diferencia entre value y una variable en scala?
//value es inmutable y var puede cambiar su valor

//6. Dada la tupla (2,4,5,1,2,3,3.1416,23) regresa el numero 3.1416
val my_tup = (2,4,5,1,2,3,3.1416,23) 
my_tup._7