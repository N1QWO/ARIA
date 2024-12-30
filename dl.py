import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Bl(nn.Module):
    
    def __init__(self, in_, out_,hid = 64,hid2 = 32,device='cpu'):
        
        super(Bl, self).__init__()
        self.device = device

        self.fc11 = nn.Linear(in_, hid)
        self.fc11_b = nn.Parameter(torch.zeros(hid))
        self.fc11_activation = nn.LeakyReLU(negative_slope=0.1)

        self.cla = nn.Linear(in_, hid)
        self.cla_b = nn.Parameter(torch.zeros(hid))

        self.fc12 = nn.Linear(in_, hid)
        self.fc12_b = nn.Parameter(torch.zeros(hid))

        self.fc3 = nn.Linear(hid, hid2)
        self.fc3_b = nn.Parameter(torch.zeros(hid2))

        self.fc4 = nn.Linear(hid2, out_)
        self.layer_norm = nn.LayerNorm(hid2).to(self.device)
        self.dropout = nn.Dropout(p=0.35)

        self.mxp = nn.MaxPool1d(kernel_size=int(hid//hid2))
        
    def short_cla_module(self, input):
        short = self.fc12(input) + self.fc12_b #128
        short2 = self.fc3(short) + self.fc3_b#32
        cust = torch.tanh(short2-torch.mean(short2))*short2 #32
        
        cust = self.layer_norm(cust)
        
        return cust
        
    
    def long_cla_module(self, input):
        long = self.fc11_activation(self.fc11(input) + self.fc11_b)
        long = self.dropout(long)
        
        cla = torch.sigmoid(self.cla(input) + self.cla_b)

        return long * cla
    

    def forward(self, input):
        al = self.long_cla_module(input)

        cust = self.short_cla_module(input)
        
        cust = self.dropout(cust)

        al = al.unsqueeze(1)  
        maxp = self.mxp(al).squeeze(1) 

        re = self.fc4(cust * maxp)
        return re

    def fit(self, X, y, batch_size, epochs, learning_rate=0.001, device='cpu'):
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        loss_history, loss_mape = [], []
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            X_shuffled, y_shuffled = X[indices].to(device), y[indices].to(device)
            
            epoch_loss, epoch_loss_mape = 0.0, 0.0
            for i in range(0, dataset_size, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                
                loss = criterion(predictions, y_batch)
                loss_data = torch.abs(y_batch - predictions) / torch.clamp(y_batch, min=1e-7)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_mape += torch.mean(loss_data).item()
            
            scheduler.step()
            loss_history.append(epoch_loss / (dataset_size // batch_size))
            loss_mape.append(epoch_loss_mape / (dataset_size // batch_size))
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_history[-1]:.10f}, Loss_mape: {loss_mape[-1]:.10f}')
        
        return loss_history, loss_mape

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: Размерность векторного представления входных данных (embedding dimension).
            num_heads: Количество голов в механизме внимания.
        """
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")

        # Линейные слои для создания Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Линейный слой для объединения голов
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Тензор входных данных формы (batch_size, seq_len, embed_dim).
        Returns:
            Тензор формы (batch_size, seq_len, embed_dim) с обработанным вниманием.
        """
        batch_size, seq_len, embed_dim = x.shape

        # Убедимся, что embed_dim соответствует инициализации
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"

        # Вычисление Q, K, V
        Q = self.query(x)  # (batch_size, seq_len, embed_dim)
        K = self.key(x)    # (batch_size, seq_len, embed_dim)
        V = self.value(x)  # (batch_size, seq_len, embed_dim)

        # Разделение на головы
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Вычисление оценок внимания (Scaled Dot-Product Attention)
        energy = torch.einsum("bhqd,bhkd->bhqk", Q, K)  # (batch_size, num_heads, seq_len, seq_len)
        scaling_factor = self.head_dim ** 0.5
        attention = F.softmax(energy / scaling_factor, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # Применение внимания к V
        out = torch.einsum("bhqk,bhvd->bhqd", attention, V)  # (batch_size, num_heads, seq_len, head_dim)

        # Объединение голов
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Пропуск через выходной линейный слой
        out = self.fc_out(out)  # (batch_size, seq_len, embed_dim)

        return out

class FeedForwardRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim: Размерность входных данных.
            hidden_dim: Размерность скрытого слоя.
            output_dim: Размерность выходного слоя (для регрессии = 1).
        """
        super(FeedForwardRegression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Args:
            x: Тензор входных данных формы (batch_size, input_dim).
        Returns:
            Тензор формы (batch_size, output_dim).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class Baseline(nn.Module):
    def __init__(self, in_dim, out_dim, embed_dim, num_heads, hidden_dim,device):
        """
        Объединенный класс, использующий Bl, SelfAttention и FeedForwardRegression.

        Args:
            in_dim: Размер входных данных для Bl.
            out_dim: Размер выходных данных для FeedForwardRegression.
            embed_dim: Размер embedding для SelfAttention.
            num_heads: Количество голов для SelfAttention.
            hidden_dim: Размерность скрытого слоя для FeedForwardRegression.
        """
        super(Baseline, self).__init__()
        self.bl = Bl(in_dim, embed_dim,device=device)
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForwardRegression(embed_dim, hidden_dim, out_dim)

    def forward(self, x):
        """
        Args:
            x: Тензор входных данных формы (batch_size, seq_len, in_dim).
        Returns:
            Тензор формы (batch_size, out_dim).
        """
        x_bl = self.bl(x)  # (batch_size, embed_dim)
        x_bl = x_bl.unsqueeze(1)  # Добавляем измерение seq_len
        x_sa = self.self_attention(x_bl)  # (batch_size, seq_len, embed_dim)
        x_sa = x_sa.squeeze(1)  # Убираем измерение seq_len
        output = self.feed_forward(x_sa)  # (batch_size, out_dim)
        return output
    
    def fit(self, X, y, batch_size, epochs, learning_rate=0.001, device='cpu'):
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        loss_history, loss_mape = [], []
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            X_shuffled, y_shuffled = X[indices].to(device), y[indices].to(device)
            
            epoch_loss, epoch_loss_mape = 0.0, 0.0
            for i in range(0, dataset_size, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                
                loss = criterion(predictions, y_batch)
                loss_data = torch.abs(y_batch - predictions) / torch.clamp(y_batch, min=1e-7)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_mape += torch.mean(loss_data).item()
            
            scheduler.step()
            loss_history.append(epoch_loss / (dataset_size // batch_size))
            loss_mape.append(epoch_loss_mape / (dataset_size // batch_size))
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss_history[-1]:.10f}, Loss_mape: {loss_mape[-1]:.10f}')
        
        return loss_history, loss_mape
    





class THdLin(nn.Module):
    
    def __init__(self, in_, out_,hid = 64,hid2 = 32,hid3=100,device='cpu'):
        
        super(THdLin, self).__init__()
        self.device = device

        self.fc11 = nn.Linear(in_, hid)
        self.fc11_b = nn.Parameter(torch.zeros(hid))

        self.fc12 = nn.Linear(hid, hid2)
        self.fc12_b = nn.Parameter(torch.zeros(hid2))


        self.dp1 = nn.Dropout(0.3)

        self.fc13 = nn.Linear(hid2, hid3)
        self.fc13_b = nn.Parameter(torch.zeros(hid3))
        self.bn =nn.LayerNorm(hid3)
        
        self.dp2 = nn.Dropout(0.3)
        self.fc14 = nn.Linear(hid3, out_)
        # self.bn2 =nn.BatchNorm1d(hid4)


        # self.fc15 = nn.Linear(hid4, hid5)
        # self.fc15_b = nn.Parameter(torch.zeros(hid5))
        # self.dp3  = nn.Dropout(0.15)
        # self.fc16 = nn.Linear(hid5, out_)
        #self.fc13_b = nn.Parameter(torch.zeros(hid3))
    

    def forward(self, input):
        f1 = self.fc11(input) + self.fc11_b
        f2 = F.relu(self.fc12(f1) + self.fc12_b)

        dp1 = self.dp1(f2)
        
        f3 = self.fc13(dp1) + self.fc13_b
        bn = self.bn(f3)

        dp2 = self.dp2(bn)
        f4 = self.fc14(dp2)
        # bn2 = self.bn2(f4)

        # f5 = F.leaky_relu(self.fc15(bn2) + self.fc15_b,negative_slope=0.1)
        # dp3 = self.dp3(f5)
        # f6 = self.fc16(dp3)

        return f4

    def fit(self, X, y,X_t,y_t, batch_size, epochs, learning_rate=0.001, device='cpu',loss_tube=5):
        self.to(device)
        criterion = nn.HuberLoss(delta=0.01)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5,eps = 1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        loss_history, loss_mape = [], []
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            X_shuffled, y_shuffled = X[indices].to(device), y[indices].to(device)
            
            epoch_loss, epoch_loss_mape = 0.0, 0.0
            for i in range(0, dataset_size, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                
                loss = torch.sqrt(criterion(predictions, y_batch))
                loss_data = torch.abs(y_batch - predictions) / torch.clamp(y_batch, min=1e-7)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_mape += torch.mean(loss_data).item()
            
            scheduler.step()
            loss_history.append(epoch_loss / (dataset_size // batch_size))
            loss_mape.append(epoch_loss_mape / (dataset_size // batch_size))
            
            if (epoch + 1) % 10 == 0:
                self.eval()
                loss_rd = torch.abs(y_t - self.forward(X_t)) / torch.clamp(y_t, min=1e-7)
                per_loss_rd = loss_rd[loss_rd<0.01*loss_tube].shape[0] / loss_rd.shape[0]
                print(f'Epoch {epoch + 1}/{epochs}, Loss_model: {loss_history[-1]:.10f}, Loss_mape_train: {loss_mape[-1]:.10f}, Loss_mape_test: {torch.mean(loss_rd).item():.10f}, tube_loss_mape_test: {per_loss_rd:.10f}')
                self.train()
        
        return loss_history, loss_mape
    


class three_layer(nn.Module):
    
    def __init__(self, in_, out_,hid = 64,hid2 = 32,hid3=100,device='cpu'):
        
        super(three_layer, self).__init__()
        self.device = device

        self.fc11 = nn.Linear(in_, hid)
        self.fc11_b = nn.Parameter(torch.zeros(hid))

        self.fc12 = nn.Linear(hid, hid2)
        self.fc12_b = nn.Parameter(torch.zeros(hid2))

        self.fc13 = nn.Linear(hid2, hid3)
        self.fc13_b = nn.Parameter(torch.zeros(hid3))
        
        self.fc14 = nn.Linear(hid3, out_) 
        self.fc14_b = nn.Parameter(torch.zeros(out_))
        # self.bn2 =nn.BatchNorm1d(hid4)


        # self.fc15 = nn.Linear(hid4, hid5)
        # self.fc15_b = nn.Parameter(torch.zeros(hid5))
        # self.dp3  = nn.Dropout(0.15)
        # self.fc16 = nn.Linear(hid5, out_)
        #self.fc13_b = nn.Parameter(torch.zeros(hid3))
    

    def forward(self, input):
        f1 = F.relu(self.fc11(input) + self.fc11_b)
        f2 = F.relu(self.fc12(f1) + self.fc12_b)

        f3 = F.relu(self.fc13(f2) + self.fc13_b)

        f4 = self.fc14(f3) + self.fc14_b
        # bn2 = self.bn2(f4)

        # f5 = F.leaky_relu(self.fc15(bn2) + self.fc15_b,negative_slope=0.1)
        # dp3 = self.dp3(f5)
        # f6 = self.fc16(dp3)

        return f4

    def fit(self, X, y,X_t,y_t, batch_size, epochs, learning_rate=0.001, device='cpu',loss_tube=5):
        self.to(device)
        criterion = nn.MSELoss()
        #optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5,eps = 1e-6)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        loss_history, loss_mape,Loss_mape_test,tube_loss_mape_test = [], [],[],[]
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            X_shuffled, y_shuffled = X[indices].to(device), y[indices].to(device)
            
            epoch_loss, epoch_loss_mape = 0.0, 0.0
            for i in range(0, dataset_size, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                
                loss = torch.sqrt(criterion(predictions, y_batch))
                loss_data = torch.abs(y_batch - predictions) / torch.clamp(y_batch, min=1e-7)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_mape += torch.mean(loss_data).item()
            
            scheduler.step()
            loss_history.append(epoch_loss / (dataset_size // batch_size))
            loss_mape.append(epoch_loss_mape / (dataset_size // batch_size))

            self.eval()
            loss_rd = torch.abs(y_t - self.forward(X_t)) / torch.clamp(y_t, min=1e-7)
            per_loss_rd = loss_rd[loss_rd<0.01*loss_tube].shape[0] / loss_rd.shape[0]

            Loss_mape_test.append(torch.mean(loss_rd).item())
            tube_loss_mape_test.append(per_loss_rd)
            self.train()
            
            if (epoch + 1) % 100 == 0 or epoch == 9:
                print(f'Epoch {epoch + 1}, mse_train: {loss_history[-1]:.6f}, mape_train: {loss_mape[-1]:.6f}, mape_test: {Loss_mape_test[-1]:.6f}, tube_mape_test: {tube_loss_mape_test[-1]:.6f}')
                
        
        return loss_history, loss_mape,Loss_mape_test,tube_loss_mape_test
    



class up_three_layer(nn.Module):
    
    def __init__(self, in_, out_,hid = 64,hid2 = 32,hid3=100,device='cpu'):
        
        super(up_three_layer, self).__init__()
        self.device = device

        self.fa = nn.LeakyReLU(negative_slope=0.1)

        self.fc11 = nn.Linear(in_, hid)
        self.fc11_b = nn.Parameter(torch.zeros(hid))

        # self.dp1 = nn.Dropout(0.05)
        # self.dp2 = nn.Dropout(0.05)
        # self.dp3 = nn.Dropout(0.05)

        self.layer_norm1 = nn.LayerNorm(hid).to(self.device)

        self.fc12 = nn.Linear(hid, hid2)
        self.fc12_b = nn.Parameter(torch.zeros(hid2))


        self.layer_norm3 = nn.LayerNorm(hid2).to(self.device)
        
        self.fc13 = nn.Linear(hid2, hid3)
        self.fc13_b = nn.Parameter(torch.zeros(hid3))
        

        self.fc14 = nn.Linear(hid3, out_) 
        self.fc14_b = nn.Parameter(torch.zeros(out_))
        # self.bn2 =nn.BatchNorm1d(hid4)


        # self.fc15 = nn.Linear(hid4, hid5)
        # self.fc15_b = nn.Parameter(torch.zeros(hid5))
        # self.dp3  = nn.Dropout(0.15)
        # self.fc16 = nn.Linear(hid5, out_)
        #self.fc13_b = nn.Parameter(torch.zeros(hid3))
    

    def forward(self, input):
        f1 =  self.fa(self.fc11(input) + self.fc11_b)
        f1 = self.layer_norm1(f1)
        #dp1 = self.dp1(f1)

        f2 = self.fa(self.fc12(f1) + self.fc12_b)
        #dp2 = self.dp2(f2) 
        
        f3 = self.layer_norm3(f2)

        f3 =  self.fa(self.fc13(f3) + self.fc13_b)
        #dp3 = self.dp3(f3)

        f4 = self.fc14(f3) + self.fc14_b
        # bn2 = self.bn2(f4)

        # f5 = F.leaky_relu(self.fc15(bn2) + self.fc15_b,negative_slope=0.1)
        # dp3 = self.dp3(f5)
        # f6 = self.fc16(dp3)

        return f4

    def fit(self, X, y,X_t,y_t, batch_size, epochs, learning_rate=0.001, device='cpu',loss_tube=5):
        self.to(device)
        criterion = nn.HuberLoss(delta = 0.01)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        loss_history, loss_mape,Loss_mape_test,tube_loss_mape_test = [], [],[],[]
        dataset_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = torch.randperm(dataset_size)
            X_shuffled, y_shuffled = X[indices].to(device), y[indices].to(device)
            
            epoch_loss, epoch_loss_mape = 0.0, 0.0
            for i in range(0, dataset_size, batch_size):
                X_batch, y_batch = X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]
                optimizer.zero_grad()
                predictions = self.forward(X_batch)
                
                loss = torch.sqrt(criterion(predictions, y_batch))
                loss_data = torch.abs(y_batch - predictions) / torch.clamp(y_batch, min=1e-7)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_loss_mape += torch.mean(loss_data).item()
            
            scheduler.step()
            loss_history.append(epoch_loss / (dataset_size // batch_size))
            loss_mape.append(epoch_loss_mape / (dataset_size // batch_size))

            self.eval()
            loss_rd = torch.abs(y_t - self.forward(X_t)) / torch.clamp(y_t, min=1e-7)
            per_loss_rd = loss_rd[loss_rd<0.01*loss_tube].shape[0] / loss_rd.shape[0]

            Loss_mape_test.append(torch.mean(loss_rd).item())
            tube_loss_mape_test.append(per_loss_rd)
            self.train()
            
            if (epoch + 1) % 100 == 0 or epoch == 9:
                print(f'Epoch {epoch + 1}, huber_train: {loss_history[-1]:.6f}, mape_train: {loss_mape[-1]:.6f}, mape_test: {Loss_mape_test[-1]:.6f}, tube_mape_test: {tube_loss_mape_test[-1]:.6f}')
                
        
        return loss_history, loss_mape,Loss_mape_test,tube_loss_mape_test