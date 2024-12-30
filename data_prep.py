import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def data_preparation(dats,feature,label,window_roll = 10,device='cpu'):
    print(dats.shape)
    dats[:,1] = scale(dats[:,1])
    f = np.roll(dats[:,1], shift=1, axis=0)

    #print(f.shape)
    dats = np.concatenate((f.reshape(-1,1),dats), axis=1)[1:]
    #print(dats.shape)
    shape = (window_roll,3)
    datset = np.lib.stride_tricks.sliding_window_view(dats, window_shape=shape)
    datset = datset.reshape(-1,datset.shape[2],datset.shape[3])
    #print(datset[:5])
    datset = data_squeeze(datset,30)
    X = torch.tensor(datset[:,:,:2]).float()
    y = torch.tensor(datset[:,:,2]).float()
    print(X.shape)
    print(y.shape)
    #X = X.reshape(-1,window_roll,feature).float().to(device)
    #y = y.reshape(-1,window_roll).float().to(device)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42)


    dataset = {'train_input': X_train,'train_label': y_train,
               'test_input' : X_test, 'test_label' : y_test}
    return dataset


def load_data(path):
    dats = []
    with open(path, 'r') as f:
        lines = f.readlines()  # Чтение строк, разделенных \n

    for line in lines:
        values = line.strip().split('\t')  # Убираем лишние пробелы и разделяем по \t
        if len(values) == 2:  # Убедимся, что в строке 2 значения
            try:
                dats.append([float(values[0]), float(values[1])])  # Преобразуем к float и добавляем в список
            except ValueError:
                print(f"Ошибка преобразования строки: {line}")  # Сообщаем, если не удалось преобразовать строку

    data = np.array(dats)  # Преобразуем список в numpy-массив
    return data

import matplotlib.pyplot as plt


def show_plot_loss(loss_train, loss_test, start_epoch=0):
    # Индексы эпох с учетом start_epoch
    epochs = range(start_epoch + 1, len(loss_train) + 1)
    
    # Обрезка массивов потерь, начиная с start_epoch
    loss_train = loss_train[start_epoch:]
    loss_test = loss_test[start_epoch:]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_train, label='Потери на обучении', color='#FFA500', marker='o', linewidth=2)
    plt.plot(epochs, loss_test, label='Потери на тестировании', color='#1F4E79', marker='o', linewidth=2)

    plt.title('Потери на обучении и тестировании по эпохам', fontsize=16)
    plt.xlabel('Эпохи', fontsize=14)
    plt.ylabel('Потери', fontsize=14)

    plt.legend(title='Обозначения', fontsize=12, title_fontsize=14)

    # Сетка
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()



def scale(data):
    return data/np.log(1/np.power(data,0.1) + data) + data

def data_squeeze(data,test_size=10):

    

    permutation = np.random.choice(data.shape[0], size=int(data.shape[0] * test_size / 100), replace=False)
    return data[permutation]


# loss_train = [2.7, 0.03, 0.02, 0.004, 0.003]
# loss_test = [1.0, 0.8, 0.6, 0.5, 0.4]
# show_plot_loss(loss_train,loss_test)

def feature_label(df,feature):
    roll  = df.rolling(window=feature)
    datset = np.array([window.values for window in roll if len(window) == feature])

    return datset
def exit_Xy(datset,feature,device,f=2):
    X,y = torch.tensor(datset),torch.tensor(datset[:,-1,-1])

    X = X.reshape(-1,X.shape[1]*X.shape[2]).float().to(device)
    X = X[:,:-1]

    y = y.reshape(-1,1).float().to(device)
    return X,y

    #X,y = torch.tensor(datset[:,:,0]),torch.tensor(datset[:,-1,1])

def ready_data(feature,device,ms=100,shufle=True):
    data = []
    with open('dataset.txt', 'r') as f:
        data = f.readlines()

    dats = []
    for x in data:
        s = x.replace('\n', '').split('\t')
        dats.append([float(s[0]),float(s[1])])

    df = pd.DataFrame(dats)
    df = df[::ms]
    #df = df[(df[0]<140) & (df[0]>25)]
    
    datset = feature_label(df,feature)



    X,y = exit_Xy(datset,feature,device)
    
    if shufle:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
    else:
        r = X.shape[0]
        X_train, X_test, y_train, y_test = X[:int(r*0.7)], X[int(r*0.7):],y[int(r*0.7)],y[int(r*0.7):]


    dataset = {'train_input': X_train,'train_label': y_train,
            'test_input': X_test,'test_label': y_test}

    return dataset,df


import os

def parse_file_to_tensor(file_path):
    """
    Читает файл, парсит данные и возвращает их как тензор.
    
    Args:
        file_path (str): Путь к файлу.
        
    Returns:
        torch.Tensor: Тензор с транспонированными данными.
    """
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            # Чтение файла, преобразование в список чисел
            data.append([float(value) for line in f for value in line.split()])
        # Создаём тензор и транспонируем в столбец
        tensor_data = torch.tensor(data, dtype=torch.float32).view(-1, 5334).transpose(1,0)
        
        return tensor_data
    except Exception as e:
        print(f"Ошибка при обработке файла {file_path}: {e}")
        return torch.tensor([], dtype=torch.float32)



def load_and_concatenate_tensors(directory_path):
    """
    Загружает данные из всех файлов в директории и объединяет их в один тензор.
    
    Args:
        directory_path (str): Путь к директории.
        
    Returns:
        torch.Tensor: Тензор, объединяющий данные из всех файлов.
    """
    final_tensor = torch.tensor([], dtype=torch.float32).view(-1, 10)  # Пустой тензор-столбец
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            # Загружаем данные из файла и конкатенируем с общим тензором
            file_tensor = parse_file_to_tensor(file_path)
            #print(file_tensor.shape)
            final_tensor = torch.cat((final_tensor, file_tensor), dim=0)
    #print('final_tensor',final_tensor.shape)
    return final_tensor

# Основной процесс
def process_directory_to_tensor(directory_path, output_file):
    """
    Обрабатывает директорию, загружает данные из файлов, объединяет их в один тензор и сохраняет.
    
    Args:
        directory_path (str): Путь к директории.
        output_file (str): Имя выходного файла для сохранения.
    """
    final_tensor = load_and_concatenate_tensors(directory_path)
    #final_tensor = final_tensor.view(-1,5334,10)
    
    if final_tensor.numel() > 0:  # Проверяем, что тензор не пустой
        torch.save(final_tensor, output_file)
    else:
        print("Данные не были загружены или директория пуста.")

def ten_ready_data(path, feature, f=2, device='cpu',scale=1):
    # Загрузка данных
    data = torch.load(path)
    print(data[0:5])
    # print('data_all',data.shape)
    data_new = data[:, 1:].reshape(-1, 5334, 9)  # Удаление первого столбца и изменение формы
    # print('data_reshape',data.shape)
    # print(data[10][0])
    if scale<1:
        print('ERR')
    if scale!=1:
        data_new = data_new[:, ::scale, :]
    # print('data_scale',data.shape)
    # print(data[20])
        
    # Обработка данных
    all_windows = []
    for i in range(data_new.shape[0]):
        df = pd.DataFrame(data_new[i])
        # Rolling окна
        roll = df.rolling(window=feature)
        windows = [window.values for window in roll if len(window) == feature]
        all_windows.extend(windows)
    
    # Преобразование в массив numpy
    n_data = np.array(all_windows)

    # Формирование X и y
    X = torch.tensor(n_data).float()
    y = torch.tensor(n_data[:, -1, -1]).float()  # Целевая переменная (последний элемент второго столбца)
    #print(y)
    # Приведение форматов X и y
    X = X.reshape(-1, feature * f).to(device)  # Изменение формы X
    X = X[:, :-1]  # Удаление последнего столбца

    y = y.reshape(-1, 1).to(device)  # Приведение y к двумерному массиву

    return X, y, pd.DataFrame(data)

def req_df(model,df,input_x):   
    predict = model(input_x)
    prev = input_x[:,9:]
    new_data = torch.tensor(df)

    new_inp = torch.cat((prev,predict,new_data),1)
    return new_inp,predict


if __name__ == "__main__":
    #process_directory_to_tensor('data','data_all')
    #load_and_concatenate_tensors('data')
    #data = torch.load('data_all')
    # data = torch.load('data_150ms')
    # # data = data[::50]
    # # torch.save(data, 'data_150ms')
    # df = pd.DataFrame(data)
    # print(df.head)
    #(35845, 10)
    #(35836, 10, 9)
    #torch.Size([35836, 89]) torch.Size([35836, 1])
    z = 5334
    #X,y,df  = ten_ready_data('data_all',10,9,scale = 50)
    df_z = torch.tensor([[1,2,3,4,5,6,7,8]])
    y = torch.tensor([[1000]])
    input_x = torch.zeros((1,10*9-1))
    print(df_z.shape,y.shape,input_x.shape)
    new_inp = torch.cat((input_x,y,df_z),1)[:,9:]
    print(new_inp.shape)
    # print(X)
    # print(X.shape,y.shape)


    #load_and_concatenate_tensors('data')