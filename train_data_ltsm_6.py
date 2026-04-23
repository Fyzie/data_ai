import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import imageio

df = pd.read_excel('/home/hafizi/Documents/Data Driven Machine Learning/sorted_data.xlsx', 
                   parse_dates=['Date'], index_col='Date')

df.reset_index(drop=True, inplace=True)

df = df[df["PROFILE"] == 'AC715891'].copy()

df = df.iloc[:, 1:]

scaler_press_temp = StandardScaler()
scaler_billet_temp = StandardScaler()
scaler_extrusion_speed = StandardScaler()

df["PROFILE_EXIT_TEMP"] = scaler_press_temp.fit_transform(df[["PROFILE_EXIT_TEMP"]])
df["BILLET_TEMP"] = scaler_billet_temp.fit_transform(df[["BILLET_TEMP"]])
df["RAM_SPEED"] = scaler_extrusion_speed.fit_transform(df[["RAM_SPEED"]])

def create_cycle_sequences(data, target_column, seq_length=5):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i + seq_length].values)
        Y.append(data.iloc[i + seq_length][target_column])
    return np.array(X), np.array(Y)

SEQ_LENGTH = 5
X, Y = create_cycle_sequences(df, target_column="RAM_SPEED", seq_length=SEQ_LENGTH)

X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

input_size = X.shape[2]
hidden_layer_size = 64
output_size = 1

model = LSTMModel(input_size, hidden_layer_size, output_size)

train_size = int(0.7 * len(X))
X_train, X_test = torch.tensor(X[:train_size], dtype=torch.float32), torch.tensor(X[train_size:], dtype=torch.float32)
y_train, y_test = torch.tensor(Y[:train_size], dtype=torch.float32), torch.tensor(Y[train_size:], dtype=torch.float32)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs} | Loss: {loss.item():.6f}')

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)
    test_loss = criterion(y_pred_test.squeeze(), y_test)
    print(f'Test Loss: {test_loss.item():.6f}')

y_pred_test_numpy = y_pred_test.numpy().reshape(-1, 1)
predicted_speed_inverse = scaler_extrusion_speed.inverse_transform(y_pred_test_numpy)

actual_extrusion_speed = X_test[:, -1, 2].numpy().reshape(-1, 1)
actual_extrusion_speed_inverse = scaler_extrusion_speed.inverse_transform(actual_extrusion_speed)

actual_press_temp = X_test[:, -1, 0].numpy().reshape(-1, 1)
actual_billet_temp = X_test[:, -1, 1].numpy().reshape(-1, 1)

actual_press_temp_inverse = scaler_press_temp.inverse_transform(actual_press_temp)
actual_billet_temp_inverse = scaler_billet_temp.inverse_transform(actual_billet_temp)

images = []

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

actual_speed_points = []
predicted_speed_points = []
press_temp_points = []
billet_temp_points = []

for i in range(len(actual_extrusion_speed_inverse)):
    actual_speed_points.append(actual_extrusion_speed_inverse[i])
    predicted_speed_points.append(predicted_speed_inverse[i])
    press_temp_points.append(actual_press_temp_inverse[i])
    billet_temp_points.append(actual_billet_temp_inverse[i])

    ax1.cla()
    ax2.cla()

    ax1.plot(press_temp_points, label='Press Exit Temperature', color='green', linestyle='-.')
    ax1.plot(billet_temp_points, label='Billet Temperature', color='purple', linestyle=':')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature')
    ax1.set_title('Real-Time Temperature Monitoring')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(actual_speed_points, label='Actual Extrusion Speed', color='blue')
    ax2.plot(predicted_speed_points, label='Predicted Extrusion Speed', color='red', linestyle='--')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Extrusion Speed')
    ax2.set_title('Real-Time Extrusion Speed Prediction')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.pause(0.1)
    time.sleep(0.1)

    fig.canvas.draw()

    image = np.array(fig.canvas.renderer.buffer_rgba())

    if len(images) == 0:
        fixed_shape = image.shape[:2]

    if image.shape[:2] != fixed_shape:
        from skimage.transform import resize
        image = resize(image, fixed_shape, anti_aliasing=True, preserve_range=True).astype(np.uint8)

    images.append(image)

imageio.mimsave("real_time_plot.gif", images, fps=10)

plt.ioff()
plt.show()
