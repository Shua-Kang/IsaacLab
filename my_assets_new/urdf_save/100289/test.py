import os

# 修改为你的目标文件夹路径
folder_path = "textured_objs"

for filename in os.listdir(folder_path):
    if filename.startswith("original_"):
        old_path = os.path.join(folder_path, filename)
        new_filename = filename.replace("original_", "original_", 1)
        new_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
