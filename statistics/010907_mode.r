# Задачка на программирование.
# 
# В R мы без труда можем рассчитать среднее и медиану вектора, а вот встроенной функции для расчета моды — наиболее часто встречаемого значения — в R нет! А мода так бы пригодилась нам при анализе номинативных данных! При этом функция mode в R существует, но выполняет абсолютно другую задачу (если хотите узнать, какую именно, ознакомьтесь со справкой: наберите в консоли ?mode).
# 
# Напишите функцию stat_mode, которая получает на вход вектор из чисел произвольной длины и возвращает числовой вектор с наиболее часто встречаемым значением. Если наиболее часто встречаемых значений несколько, функция должна возвращать несколько значений моды  в виде числового вектора. 
# 
# > v <- c(1, 2, 3, 3, 3, 4, 5)
# > stat_mode(v)
# [1] 3
# 
# > v <- c(1, 1, 1, 2, 3, 3, 3)
# > stat_mode(v)
# [1] 1 3


stat_mode <- function (v) {
  t <- table(v)
  w <- which(t == max(t))
  return(as.integer(names(w)))
}

