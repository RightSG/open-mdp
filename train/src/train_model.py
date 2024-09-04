import csv
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 定义数据集路径
dataset_path = "datasets/"

# 定义 CSV 文件路径
csv_file_path = "combined_index.csv"

# 初始化一个列表来存储所有的 JSON 数据
json_data_list = []

# 读取 CSV 文件
with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        # 获取 JSON 文件路径
        json_file_path = os.path.join(dataset_path, row["file_name"])

        # 读取 JSON 文件
        with open(json_file_path, mode="r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
            # 将 JSON 数据添加到列表中
            json_data_list.append(
                {
                    "song_id": row["song_id"],
                    "level_index": row["level_index"],
                    "fit_diff": row["fit_diff"],
                    "ds": row["ds"],
                    "combined_diff": float(row["combined_diff"]),
                    "data": json_data,
                }
            )

# 打印读取的 JSON 数据数量
print(f"读取了 {len(json_data_list)} 个 JSON 文件的数据")

# 定义 NoteType 和 TouchArea 的映射
note_type_mapping = {"Tap": 0, "Slide": 1, "Hold": 2, "Touch": 3, "TouchHold": 4}
touch_area_mapping = {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5}


# 定义自定义数据集类
class NoteDataset(Dataset):
    def __init__(self, json_data_list):
        self.data = []
        self.labels = []
        for item in json_data_list:
            notes_sequence = []
            for entry in item["data"]:
                for note in entry["Notes"]:
                    note_features = [
                        note["holdTime"],
                        int(note["isBreak"]),
                        int(note["isEx"]),
                        int(note["isFakeRotate"]),
                        int(note["isForceStar"]),
                        int(note["isHanabi"]),
                        int(note["isSlideBreak"]),
                        int(note["isSlideNoHead"]),
                        note_type_mapping[note["noteType"]],
                        note["slideStartTime"],
                        note["slideTime"],
                        note["startPosition"],
                        touch_area_mapping[note["touchArea"]],
                        note["time"],
                        note["density"],
                        note["sweepAllowed"],
                        note["multiPressCount"],
                        note["displacement"],
                    ]
                    notes_sequence.append(note_features)
            self.data.append(notes_sequence)
            self.labels.append(item["combined_diff"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(
            self.labels[idx], dtype=torch.float32
        )


# 创建数据集
note_dataset = NoteDataset(json_data_list)

# 计算数据集中最大 note 数量
max_note_length = max(len(sequence) for sequence in note_dataset.data)
print(f"数据集中最大 note 数量: {max_note_length}")

# 使用训练集中最大 note 数量 + 200 作为固定长度，一般地，这个值通常在 1600 左右
fixed_length = max_note_length + 200


# 自定义 collate_fn 函数
def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True)
    # 如果序列长度超过固定长度，则截断；否则填充
    if data_padded.size(1) > fixed_length:
        data_padded = data_padded[:, :fixed_length, :]
    else:
        padding_size = fixed_length - data_padded.size(1)
        data_padded = torch.nn.functional.pad(data_padded, (0, 0, 0, padding_size))
    labels = torch.stack(labels)
    return data_padded, labels


# 创建数据加载器
data_loader = DataLoader(
    note_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)


# 定义注意力层
class Attention(nn.Module):
    def __init__(self, hidden_size, special_indices, special_weight=10.0):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)
        self.special_indices = special_indices
        self.special_weight = special_weight

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size]
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = self.context_vector(attention_weights).squeeze(-1)

        # 增加对特定参数的关注
        special_attention = lstm_output[:, :, self.special_indices].sum(dim=-1)
        attention_weights += self.special_weight * special_attention

        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)
        return context_vector, attention_weights


# 定义 LSTM 模型
class LSTMModelWithAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        special_indices,
        special_weight=10.0,
    ):
        super(LSTMModelWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size, special_indices, special_weight)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        lstm_output, _ = self.lstm(x, (h_0, c_0))
        context_vector, attention_weights = self.attention(lstm_output)
        output = self.fc(context_vector).squeeze(-1)  # 调整输出尺寸
        return output, attention_weights


# 定义模型参数
input_size = 18  # 输入特征的维度
hidden_size = 128  # LSTM 隐藏层的大小
num_layers = 2  # LSTM 层数
output_size = 1  # 输出的维度

# 特定参数的索引
special_indices = [
    13,
    14,
    15,
    16,
    17,
]  # 对应 note["time"], note["density"], note["sweepAllowed"], note["multiPressCount"], note["displacement"]

# 创建模型
model = LSTMModelWithAttention(
    input_size,
    hidden_size,
    num_layers,
    output_size,
    special_indices,
    special_weight=10.0,
).cuda()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=5, factor=0.5
)

# 定义早停机制
early_stopping_patience = 10
best_loss = float("inf")
epochs_no_improve = 0

# 定义目标损失值
target_loss = 0.01

# 检查是否存在已保存的模型
model_path = "trained_models/best_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"加载已保存的模型: {model_path}")
else:
    print("未找到已保存的模型，从头开始训练")

# 训练模型
num_epochs = 1000  # 设置一个较大的初始值
save_interval = 10  # 每10轮保存一次模型

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs, attention_weights = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(data_loader)
    scheduler.step(epoch_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 保存模型
    if (epoch + 1) % save_interval == 0:
        model_save_path = os.path.join("trained_models", f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存: {model_save_path}")

    # 检查当前损失值并与 best 比对
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        epochs_no_improve = 0
        # 保存最好的模型
        best_model_path = "trained_models/best_model.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"最好的模型已保存: {best_model_path}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print("早停机制触发，停止训练")
        break

    # 检查是否达到目标损失值
    if epoch_loss <= target_loss:
        print(f"达到目标损失值 {target_loss}，停止训练")
        break

print("训练完成")
