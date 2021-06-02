# Учебный проект "multilayer_perceptron"

## Архитектура проекта
Изначально планировал сделать имплементацию многослойного перцептрона Розенблатта по [заданию в школе 21](docs/subject.pdf) с архитектурой схожей с sklearn.neural_network.MLPClassifier, 
однако проект перерос в нечто большее - мини имплементацию pytorch/tensorflow, написанную на Python 3.8.

## Присутствующие блоки
**Слои:**  
* Полносвязный слой (Dense)  
* Dropout
  
**Активации:**  
* ReLU  
* Tanh  
* Sigmoid  
* Softmax  
  
**Функции потерь:**  
* Кросс-энтропия
* Бинарная кросс-энтропия
* MeanSquaredError  

**Оптимизаторы:**
* SGD
* Momentum
* RMSProp
* Adam

**Другое:**
* Инициализации весов (xavier_normal, kaiming_normal)
* Клиппирование градиента
* EarlyStopping  

## В процессе добавления:

* Сверточный слой
* MaxPooling
* Batch normalization

## Замеры точности

Датасеты были разделены на train/val/test выборки. Замеры точности были проведены на test выборке.
При желании вы можете запустить заготовленные тесты в папке tests и проверить адекватность моделей.

### Классификация

Метрика - accuracy

Датасет | LogisticRegression (sklearn) | MLPClassifier (sklearn) | Моя имплементация перцептрона
--- | --- | --- | ---
breast cancer | 0.9681 | 0.9681 | 0.9734 |
fisher's iris | 1.0 | 1.0 | 1.0 |
MNIST | 0.9164 |0.9465 | 0.951 |

### Регрессия

Метрика - MSE

Датасет | LinearRegression (sklearn) | MLPClassifier (sklearn) | Моя имплементация перцептрона
--- | --- | --- | ---
boston | 21.0324 | 12.6515 | 11.8298 |

## Установка
>python3.8 -m venv env  
>. ./env/activate  
>pip install -r requirements.txt

## Использованные библиотеки
numpy>=1.20.1  
pandas>=1.2.3  
seaborn>=0.11.1  
matplotlib>=3.4.1  
scikit-learn>=0.24.1  
mnist>=0.2.2  
