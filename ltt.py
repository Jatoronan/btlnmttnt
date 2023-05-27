import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Bước 1: Đọc dữ liệu từ file Excel:
file_excel = "dienthoai.xlsx"
df = pd.read_excel('dienthoai.xlsx',header=1)

# Bước 2: Chuẩn bị dữ liệu:
print(df.columns)
df = df.dropna()

thuoc_tinh_quan_trong = [
    
    'Tên sản phẩm',
    'Dòng sản phẩm',
    'Phiên bản',
    'Kích thước màn hình',
    'Hệ điều hành',
    'CPU',
    'RAM',
    'Dung lượng bộ nhớ',
    'Năm phát hành',
    'Dung lượng Pin',
    'Nơi bán',
    'Kho bán',
    'Tổng lượt đánh giá',
    'Đánh giá năm sao',
    'Đã bán',
    'Giá tiền'
]

df_filtered = df[thuoc_tinh_quan_trong]
print(df_filtered)
non_numeric_columns = ['Tên sản phẩm', 'Dòng sản phẩm']
numeric_columns = [col for col in thuoc_tinh_quan_trong if col not in non_numeric_columns]

df_numeric = df[numeric_columns]

# Chuyển đổi cột chứa dữ liệu không phải số thành dữ liệu số bằng cách sử dụng giá trị 0 (hoặc một giá trị thích hợp khác)
for column in df_numeric.columns:
    df_numeric[column] = pd.to_numeric(df_numeric[column], errors='coerce')
    df_numeric[column].fillna(0, inplace=True)

# Bước 3: Sử dụng one-hot encoding cho cột không phải số
df_non_numeric = df[non_numeric_columns]
df_one_hot_encoded = pd.get_dummies(df_non_numeric)

# Bước 4: Kết hợp lại DataFrames đã được tính toán và chuẩn hóa dữ liệu
df_filtered = pd.concat([df_numeric, df_one_hot_encoded], axis=1)
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df_filtered)

# Bước 5: Xây dựng thuật toán cosine similarity
cosine_sim = cosine_similarity(df_normalized)

# Bước 6: Định nghĩa hàm gợi ý sản phẩm
def suggest_products(product_id, top_n=5):
    similarity_scores = list(enumerate(cosine_sim[product_id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    product_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
    suggested_products = df.iloc[product_indices].copy()
    suggested_products["similarity_score"] = [i[1] for i in similarity_scores[1:top_n + 1]]
    return suggested_products

# Hàm tìm kiếm sản phẩm theo từ khóa
def search_products(keyword):
    keyword = keyword.lower()
    indices = []
    for index, row in df.iterrows():
        if keyword in row['Tên sản phẩm'].lower() or keyword in row['Dòng sản phẩm'].lower():
            indices.append(index)
    return indices

# Hàm cho người dùng nhập từ khóa và hiển thị sản phẩm tương tự
def find_similar_products_by_keyword():
    keyword = input("Vui lòng nhập từ khóa tìm kiếm sản phẩm: ")
    product_indices = search_products(keyword)
    if not product_indices:
        print("Không tìm thấy sản phẩm phù hợp với từ khóa.")
    else:
        for index in product_indices:
            print(f"Sản phẩm tương tự nhất với sản phẩm {index}:")
            print(df.iloc[index][['Tên sản phẩm', 'Dòng sản phẩm']])
            print("Top sản phẩm tương tự:")
            suggested_products_result = suggest_products(index)
            print(suggested_products_result[['Tên sản phẩm', 'Dòng sản phẩm', 'similarity_score']])
            print("\n")

# Gọi hàm để nhập từ khóa và hiển thị sản phẩm tương tự
find_similar_products_by_keyword()
