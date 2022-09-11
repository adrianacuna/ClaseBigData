// 1.Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"
var lista = List("Rojo", "Blanco", "Negro")
lista

//2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
lista.++=(List("verde", "amarillo", "azul","naranja","perla"))
println(lista)