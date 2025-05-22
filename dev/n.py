import os
import shutil

# 入力フォルダと出力フォルダを指定
input_folder = 'dev/cube_images'
output_folder = 'dev/option'

# 出力フォルダがなければ作成
os.makedirs(output_folder, exist_ok=True)

# 入力フォルダ内のファイルを取得
files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

# ソート（ファイル名順）
files.sort()

# 各ファイルを連番付きでコピー＆リネーム
for i, filename in enumerate(files):
    _, ext = os.path.splitext(filename)
    new_name = f"{i}{ext}"
    src = os.path.join(input_folder, filename)
    dst = os.path.join(output_folder, new_name)
    shutil.copy2(src, dst)  # コピーしてメタデータ保持（必要なければ shutil.copy でも可）

print("連番ファイルを出力フォルダに保存しました。")
