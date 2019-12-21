# В лекциях я говорил, что иногда возникает необходимость перекодировать количественную переменную в номинативную. Однако зачастую мы можем создавать новую номинативную переменную, комбинируя значения нескольких количественных переменных. Рассмотрим такой пример.
# 
# Воспользуемся встроенными в R данными Iris. Они сразу доступны для работы. Если вы не знакомы с историей этого набора данных, вызовите справку:
# 
# ?iris 
# Создайте новую переменную important_cases - фактор с двумя градациями ("No" и "Yes"). Переменная должна принимать значение Yes, если для данного цветка значения хотя бы трех количественных переменных выше среднего. В противном случае переменная important_cases  будет принимать значение No.
# 
# Например, рассмотрим первую строчку данных iris:
# 
#   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
# 1          5.1         3.5          1.4         0.2  setosa
# 
# 
# В данном случае только значение  Sepal.Width 3.5 больше, чем среднее значение mean(Sepal.Width) = 3.057. Соответственно для первого цветка значение переменной important_cases будет "No".
# 
# Теперь рассмотрим 62 строчку данных
# 
#    Sepal.Length Sepal.Width Petal.Length Petal.Width    Species
# 62          5.9           3          4.2         1.5 versicolor
# 
# 
# В данном случае и значение Sepal.Length 5.9 больше чем среднее по выборке, mean(Sepal.Length)  = 5.84. Также значение Petal.Length и Petal.Width для этого цветка больше чем соответствующие средние значения: mean(Petal.Length) = 3.76,   mean(Petal.Width ) = 1.1.  Следовательно, для этого цветка значение переменной important_cases будет "Yes".
# Таким образом, если хотя бы три переменные превышают среднее значение по выборке, тогда  значение переменной important_cases будет "Yes".
# 
# Что должно получиться:
# 
# > str(iris$important_cases)
#  Factor w/ 2 levels "No","Yes": 1 1 1 1 1 1 1 1 1 1 ...
# 
# > table(iris$important_cases)
#  No Yes 
#  81  69 
# Формат ответа: в поле для ответа напишите скрипт, который создает новую переменную - фактор в данных iris. Код для проверки задания считает переменную  important_cases из данных Iris и сравнит ее с верным ответом.


cnames <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

cmeans = colMeans(iris[cnames])
get_important_value <- function(r) {
  if (sum(r[cnames] > cmeans) >= 3) {
    return("Yes")
  }
  return("No")
}

important <- apply(iris, 1, get_important_value)
iris$important_cases = as.factor(important)

# Etalon soolution
# importance_calc <- function(v1, v2, threshold=3){
#     ifelse(sum(v1 > v2) >= threshold, 'Yes', 'No')}
# iris$important_cases <- factor(apply(iris[1:4], 1, importance_calc, v2 = colMeans(iris[, 1:4])))
