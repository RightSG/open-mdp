from src.enhance_json_with_features import process_json_file
from src.inference import predict_difficulty

# 在此处修改要处理的文件路径
json_file_path = "pre-rawchart.json"


def main():
    process_json_file(json_file_path, "chart.json")

    file_name, predicted_difficulty = predict_difficulty("chart.json")
    print(f"文件名: {file_name}")
    print(f"预测的难度: {predicted_difficulty}")


if __name__ == "__main__":
    main()
