# Import các thư viện cần thiết
from idlelib.iomenu import errors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
file_path = 'data/VN_housing_dataset.csv'  # Ensure the correct path
data = pd.read_csv(file_path)

# Loại bỏ các ký tự không phải số (ví dụ: "triệu/m2") và khoảng trắng trong cột "Giá/m2"
data['Giá/m2'] = data['Giá/m2'].replace('[^\d.]', '', regex=True)
#print(data['Giá/m2'].head())
data['Diện tích'] = data['Diện tích'].replace('[^\d.]', '', regex=True)
data['Số phòng ngủ'] = data['Số phòng ngủ'].replace('[^\d.]', '', regex=True)
# Chuyển cột "Giá/m2" sang kiểu số sau khi làm sạch dữ liệu
data['Giá/m2'] = pd.to_numeric(data['Giá/m2'], errors='coerce')
data['Diện tích'] = pd.to_numeric(data['Diện tích'], errors='coerce')
data['Số phòng ngủ'] = pd.to_numeric(data['Số phòng ngủ'], errors='coerce')
# Điền giá trị thiếu (NaN) bằng giá trị trung bình
data['Giá/m2'] = data['Giá/m2'].fillna(data['Giá/m2'].mean())
data['Số phòng ngủ'] = data['Số phòng ngủ'].fillna(data['Số phòng ngủ'].mean())
# Loại bỏ các giá trị ngoại lai (giới hạn giá trị lớn nhất là 99th percentile)
upper_limit = data['Giá/m2'].quantile(0.99)
data = data[data['Giá/m2'] <= upper_limit]

# Vẽ lại biểu đồ phân phối
plt.figure(figsize=(10, 6))
sns.histplot(data['Giá/m2'], bins=30, kde=True)
plt.title('Phân phối giá nhà')
plt.xlabel('Giá (triệu VNĐ/m2)')
# Đặt các giá trị trục x dưới dạng float thay vì khoa học
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.show()



# Chỉ giữ các cột số cho ma trận tương quan
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Ma trận tương quan giữa các cột
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan')
plt.show()

# Loại bỏ các cột không cần thiết và không phải số
data = data.drop(columns=['Ngày', 'Địa chỉ', 'Quận', 'Giấy tờ pháp lý', 'Loại hình nhà ở'])

# Chuyển các cột kiểu chuỗi thành mã hóa one-hot
# Xác định tất cả các cột chứa giá trị không phải số
categorical_columns = data.select_dtypes(include=['object']).columns

# Áp dụng mã hóa one-hot cho tất cả các cột dạng chuỗi
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Tách các cột đầu vào (X) và biến mục tiêu (y)
X = data.drop(columns=['Giá/m2'])
y = data['Giá/m2']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa các cột số
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Huấn luyện mô hình Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# In các chỉ số hiệu suất
print("Lỗi bình phương trung bình:", mean_squared_error(y_test, y_pred))
print("Lỗi tuyệt đối trung bình:", mean_absolute_error(y_test, y_pred))
print("Điểm R bình phương:", r2_score(y_test, y_pred))

# Biểu đồ so sánh giá thực tế và dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Giá thực tế")
plt.ylabel("Giá dự đoán")
plt.title("So sánh giá thực tế và giá dự đoán")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# Nhập điều kiện cho trước để dự đoán
print("\nNhập điều kiện cho nhà cần dự đoán:")
area = float(input("Diện tích (m2): "))
bedrooms = int(input("Số phòng ngủ: "))
price = int(input("Số tầng: "))
huyen = str(input("Huyện : "))

# Chuyển đổi điều kiện thành mảng và chuẩn hóa
input_data = pd.DataFrame([[area, bedrooms, price, huyen]], columns=['Diện tích', 'Phòng ngủ', 'Số tầng', 'Huyện'])
input_data = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)  # Khớp với các cột đã mã hóa
input_data = scaler.transform(input_data)

# Dự đoán giá nhà
predicted_price = model.predict(input_data)
print(f"\nGiá dự đoán cho căn nhà là: {predicted_price[0]:,.2f} triệu VNĐ/m2")
