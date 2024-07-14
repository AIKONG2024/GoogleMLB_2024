import pandas as pd

# CSV 파일을 읽어옵니다
file_path = './health_insurance/data/train.csv'  # 여기서 'your_file.csv'를 실제 파일 경로로 변경하십시오
df = pd.read_csv(file_path)

# 데이터 프레임의 길이를 계산합니다
total_rows = len(df)

# 마지막 10%의 행 수를 계산합니다
num_rows_to_remove = int(total_rows * 0.5)

# 마지막 10%를 제거한 새로운 데이터 프레임을 만듭니다
df_trimmed = df.iloc[:-num_rows_to_remove]

# 결과를 확인합니다
print(df_trimmed)

# 필요한 경우, 결과를 새로운 CSV 파일로 저장합니다
output_file_path = './health_insurance/data/train_s.csv'
df_trimmed.to_csv(output_file_path, index=False)