# Учебный проект "multilayer_perceptron"

## Архитектура проекта
Изначально планировал сделать имплементацию многослойного перцептрона Розенблатта по [заданию в школе 21](docs/subject.pdf) с архитектурой схожей с sklearn.neural_network.MLPClassifier, 
однако проект перерос в нечто большее - мини имплементацию pytorch/tensorflow, написанную на Python 3.8.

## Присутствующие блоки
**Слои:**  
* Полносвязный слой (Dense)
* Conv2D
* MaxPooling2D
* Flatten
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
* Метрики (не sklearn) accuracy, precision, recall, f1_score (binary/micro/macro)

## В процессе добавления:

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


## Список выполненной работы

* Добавлен линейный слой
* Добавлены функции потерь MSE, BinaryCrossEntropy, CrossEntropy
* Добавлены функции активаций Tanh, ReLU, Sigmoid, SoftMax
* Добавлен даталоадер
* Добавлена инициализация весов (kaiming, xavier)
* Добавлены оптимизаторы SGD, Momentum, RMSProp, Adam
* Добавлен инвертированый Dropout
* Добавлен EarlyStopping
* Добавлены тесты на датасетах boston, breast cancer, fisher's iris, MNIST
* Добавлен слой Flatten и модульные тесты к нему
* Добавлен слой MaxPooling2D и модульные тесты к нему
* Добавлен слой Conv2D
* Добавлены метрики accuracy, precision, recall, f1 (binary/micro/macro) и модульные тесты к ним

## Использованные библиотеки
numpy>=1.20.1  
pandas>=1.2.3  
seaborn>=0.11.1  
matplotlib>=3.4.1  
scikit-learn>=0.24.1  
mnist>=0.2.2  
