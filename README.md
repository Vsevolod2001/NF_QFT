# NF_QFT
В квантовой механике, квантовой теории поля и статистической физике широко применяется формализм функционального интеграла. 
В виде функционального интеграла можно выразить средние по распределению Гиббса от различных наблюдаемых. 
При достаточно малой температуре они приближённо равны среднему по основному состоянию. 
	
Однако, аналитически вычислить функциональный интеграл возможно лишь в некоторых точно решаемых моделях. 
В общем случае приходится использовать различные приближённые методы, например, вычисления на решётке. 
Функциональный интеграл аппроксимируется многократным интегралом большой ($2^4-2^{10}$) размерности. 
Многократные интегралы вычисляются далее с помощью метода Монте-Карло. 
Для этого необходимо генерировать многомерные векторы (траектории), имеющие заданное распределение.

Здесь представлен нейросетевой алгоритм генерации (нормализующий поток) траекторий для вычисления функционального интеграла на решётке.

Генерируются многомерные векторы с плотностью распределения 
$$P(x)=\frac{\exp\left(-S(x)\right) }{Z}$$

Функция $S(x)$ есть евклидово действие, её вид зависит от конкретной задачи и определяется функцией гамильтона $H=T(p)+V(x)$. 
Например:
$$H=\frac{p^2}{2}+V(x) \qquad S=\sum_{i=1}^N \left[\frac{\left(x_{i+1}-x_i\right)^2}{2a}+aV(x_i)\right]$$
$$\mbox{осциллятор:} \qquad V(x)=\frac{x^2}{2}$$


Классы физических систем находятся в папке systems. Все они являются наследниками класса system. 
Классы, необходимые для построения нормализующего потока с многомасштабной архитектурой находятся в папке flows
