# Учебный проект "multilayer_perceptron"

## Архитектура проекта
Изначально планировал сделать имплементацию многослойного перцептрона Розенблатта с архитектурой схожей с sklearn.neural_network.MLPClassifier, 
однако блочная архитектура явно является более выигрышной, поэтому проект перерос в мини имплементацию pytorch/tensorflow, написанную на Python 3.8.

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
* MaxPool
* Batch normalization

## Установка
>python3.8 -m venv env  
>. ./env/activate  
>pip install -r requirements.txt  

## Запуск
>python tests/test_fcn.py

## Использованные библиотеки
numpy>=1.20.1  
pandas>=1.2.3  
seaborn>=0.11.1  
matplotlib>=3.4.1  
scikit-learn>=0.24.1  
