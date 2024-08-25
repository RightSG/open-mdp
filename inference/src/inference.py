import torch
import torch.nn as nn
import json

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('找到 %d 个可用的 GPU.' % torch.cuda.device_count())
    print('将会使用这些 GPU 推理:', torch.cuda.get_device_name(0))
else:
    print('没有可用的GPU, 将使用 CPU 推理.')
    device = torch.device("cpu")

# 定义 NoteType 和 TouchArea 的映射
note_type_mapping = {"Tap": 0, "Slide": 1, "Hold": 2, "Touch": 3, "TouchHold": 4}
touch_area_mapping = {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

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
        output = self.fc(context_vector).squeeze(-1)
        return output, attention_weights


# 创建模型
model = LSTMModelWithAttention(
    input_size,
    hidden_size,
    num_layers,
    output_size,
    special_indices,
    special_weight=10.0,
).to(device)

# 加载训练好的模型
best_model_path = "trained_models/lstm_model.pth"
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()


# 读取并处理要预测的 JSON 文件
def process_json(json_file_path):
    with open(json_file_path, mode="r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
        notes_sequence = []
        for entry in json_data:
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
        return torch.tensor(notes_sequence, dtype=torch.float32).unsqueeze(0).to(device)


# 预测函数
def predict_difficulty(json_file_path):
    input_data = process_json(json_file_path)
    with torch.no_grad():
        output, _ = model(input_data)
    return json_file_path, output.item()
