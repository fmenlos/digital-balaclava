#!/bin/sh
cd detector

# Filtrados
# 1.- elimina cabecera con "Cargadas"
# 2.- elimina lines con "patron.png" => son los patrones directamente obtenidos del procesador, esos nunca contendrán una cara
# 3.- ordena por valor numerico del campo de similitud
# 4.- conserva solo el primero de los grupos con la misma puntuacion
#cat validation$1.csv | grep -v Cargadas | grep -v patron.png | sort -t, -k3 | awk -F',' '!seen[$3]++' > validation-filtrado$1.csv
cat validation$1.csv | grep -v Cargadas | grep -v patron.png | sort -t, -k3  > validation-filtrado$1.csv


#separar las lineas que corresponden a las caras con el fondo
#separar tambien por distancias y solo los aciertos
#el clasificador nos devuelve otras imagenes que "podrían" ser matches => las descartamos
cat validation-filtrado$1.csv | grep o.png | grep 1\.00- | grep -E '([0-9]{6}\.png).*\1' | sort -t, -k1  > validation-filtrado-solocarafondo-$1-dst100.csv
cat validation-filtrado$1.csv | grep o.png | grep 0\.65- | grep -E '([0-9]{6}\.png).*\1' | sort -t, -k1  > validation-filtrado-solocarafondo-$1-dst065.csv
cat validation-filtrado$1.csv | grep o.png | grep 0\.30- | grep -E '([0-9]{6}\.png).*\1' | sort -t, -k1  > validation-filtrado-solocarafondo-$1-dst030.csv

cat validation-filtrado$1.csv | grep -v o.png | grep 1\.00- | sort -t, -k1  > validation-filtrado-conpatron-$1-dst100.csv
cat validation-filtrado$1.csv | grep -v o.png | grep 0\.65- | sort -t, -k1  > validation-filtrado-conpatron-$1-dst065.csv
cat validation-filtrado$1.csv | grep -v o.png | grep 0\.30- | sort -t, -k1  > validation-filtrado-conpatron-$1-dst030.csv



#borrar un temporal
rm validation-filtrado$1.csv


cd ..
