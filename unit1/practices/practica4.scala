// 1.Crea una lista llamada "lista" con los elementos "rojo", "blanco", "negro"
var lista = List("Rojo", "Blanco", "Negro")
lista

//2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
lista.++=(List("verde", "amarillo", "azul","naranja","perla"))
println(lista)

// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"
lista(3)
lista(4)
lista(5)

// 4. Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5
Array.range(1, 10000, 5)

//5. Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversion a conjuntos
val lista= List(1,3,3,4,6,7,3,7)
val uniq = lista.distinct
lista.toSet

// 6. Crea una mapa mutable llamado nombres que contenga los siguiente 
//"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"

val nombres = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana",27))
//  6 a . Imprime todas la llaves del mapa
nombres.keys
//  6 b . Agrega el siguiente valor al mapa("Miguel", 23)
nombres += ("Miguel" -> 23)
nombres.values
