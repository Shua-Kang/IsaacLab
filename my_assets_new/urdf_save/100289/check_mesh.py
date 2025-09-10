import trimesh

def check_mesh(filepath):
    mesh = trimesh.load_mesh(filepath)
    print(f"Checking: {filepath}")
    if not mesh.is_watertight:
        print("❌ Not watertight (may cause PhysX issues)")
    if mesh.is_empty:
        print("❌ Mesh is empty!")
    if not mesh.is_volume:
        print("⚠️ Not a volume (open surface?)")
    print("✅ Faces:", len(mesh.faces), "Vertices:", len(mesh.vertices))

# 示例：检查多个 obj 文件
from pathlib import Path

mesh_dir = Path("/p/langdiffuse/tacsl_git/tacsl/IsaacGymEnvs/assets/furniture_bench/assets/furniture/mesh/lamp")
for mesh_file in mesh_dir.glob("*.obj"):
    check_mesh(mesh_file)