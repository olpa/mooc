# Обобщим предыдущую задачу! Напишем функцию get_important_cases, которая принимает на вход dataframe с произвольным числом количественных переменных (гарантируется хотя бы две переменные). Функция должна возвращать dataframe с новой переменной - фактором important_cases.
# 
# Переменная  important_cases принимает значение Yes, если для данного наблюдения больше половины количественных переменных имеют значения больше среднего. В противном случае переменная important_cases принимает значение No.
# 
# Переменная  important_cases - фактор с двумя уровнями 0 - "No", 1  - "Yes".  То есть даже если в каком-то из тестов все наблюдения получили значения "No", фактор должен иметь две градации. 
# 
# Я написал об этой важной особенности факторов в небольшой памятке. 
# 
# Пример работы функции. 
# 
# 
# > test_data <- data.frame(V1 = c(16, 21, 18), 
#                           V2 = c(17, 7, 16), 
#                           V3 = c(25, 23, 27), 
#                           V4 = c(20, 22, 18), 
#                           V5 = c(16, 17, 19))
# 
# 
# 
# > get_important_cases(test_data)
#   V1 V2 V3 V4 V5 important_cases
# 1 16 17 25 20 16              No
# 2 21  7 23 22 17              No
# 3 18 16 27 18 19             Yes
# 



importance_calc <- function(r, means, threshold) {
  ifelse(sum(r > means) > threshold, 'Yes', 'No')
}

get_important_cases <- function(frame) {
  threshold <- dim(frame)[2] / 2
  raw_cases = apply(frame, 1, importance_calc, means = colMeans(frame), threshold = threshold)
  frame$important_cases <- factor(raw_cases, levels = c('Yes', 'No'))
  return(frame)
}


test_data <- as.data.frame(list(V1 = c(17, 21, 24, 11, 22, 22, 17, 18, 25, 26), V2 = c(20, 31, 20, 19, 27, 22, 21, 27, 19, 10), V3 = c(29, 14, 23, 24, 23, 19, 20, 24, 19, 24), V4 = c(18, 20, 23, 20, 25, 23, 18, 21, 19, 25)))
