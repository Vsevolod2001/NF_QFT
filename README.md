# NF_QFT
В квантовой механике, квантовой теории поля и статистической физике широко применяется формализм функционального интеграла. 
В виде функционального интеграла можно выразить средние по распределению Гиббса от различных наблюдаемых. 
При достаточно малой температуре они приближённо равны среднему по основному состоянию. 
	
Однако, аналитически вычислить функциональный интеграл возможно лишь в некоторых точно решаемых моделях. 
В общем случае приходится использовать различные приближённые методы, например, вычисления на решётке. 
Функциональный интеграл аппроксимируется многократным интегралом большой ($2^4-2^{10}$) размерности. 
Многократные интегралы вычисляются далее с помощью метода Монте-Карло. 
Для этого необходимо генерировать многомерные векторы (траектории), имеющие заданное распределение.

В данной работе представлен нейросетевой алгоритм генерации (нормализующий поток) траекторий для вычисления функционального интеграла на решётке.

Генерируются многомерные векторы с плотностью распределения 
$$P(x)=\frac{\exp\left(-S(x)\right) }{Z}$$

Функция $S(x)$ есть евклидово действие, её вид зависит от конкретной задачи и определяется функцией гамильтона $H=T(p)+V(x)$. 
Например:
$$H=\frac{p^2}{2}+V(x) \qquad S=\sum_{i=1}^N \left[\frac{\left(x_{i+1}-x_i\right)^2}{2a}+aV(x_i)\right]$$
$$H=|p|+V(x) \qquad S=\sum_{i=1}^{N}\left[ \ln \left(1+\frac{\left(x_{i+1}-x_i\right)^2}{a^2}\right)+aV(x_i)\right]$$
$$H=\sqrt{p^2+1}+V(x)$$
$$S=\sum_{i=1}^N\left[\ln \left(y_iK_1\left(ay_i\right)\right)+aV(x_i) \right]   \qquad y_i=1+\frac{\left(x_{i+1}-x_i\right)^2}{a^2} $$
$K_1$-функция Макдональда
$$\mbox{осциллятор:} \qquad V(x)=\frac{x^2}{2}$$





Необходимые классы для инициализации и обучения модели определены в блокноте nf_model.ipynb. 

Этот блокнот подключает файлы:
constants.py - определены параметры решётки: число узлов (N_nod), которое задаётся как некоторая степень двойки, обратная температура (Beta)
LOSS.py - определён класс функций потерь KL_with_S
Data.py - определён тренировочный датасет и даталоудер
NFandist.py - определна функция, возвращающая ортогональную матрицу заданного размера, а также другие функии, реализующие теоретические значения величин гармонического осциллятора


Классы физических систем находятся в папке systems. Все они являются наследниками класса system. 
Для запуска блокнота необходимо также импортировать соответствующую систему. (как это показано в example1 и example2).

Папка weights содержит веса обученных моделей для разных физических систем и решёток. Эти модели могут быть использованы, как показано в примерах. Перед использованием следует установить параметры N_nod и Beta в файле constants.py равными соответствущим параметрам модели.

Все файлы с классами физических систем подключают LOSS.py. Гармонический осциллятор подключает также NFandist.py.
Файлы релятивистских моделей подключают дополнительный файл: 
bessel.py - содержит собственную реализацию функции Макдональда, реализация в torch неправильно вычисляет градиент функции при малых значениях аргумента.

Релятивистский осциллятор подключает также файл
dimensionlesser.py - содержит функции, осуществляющие переход к безразмерным переменным
