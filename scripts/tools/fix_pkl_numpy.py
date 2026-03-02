import joblib
import numpy as np
import os

# 重新序列化所有 pkl 文件
src_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "motions", "e1_lab")
)

for fname in os.listdir(src_dir):
    if not fname.endswith(".pkl"):
        continue
    fpath = os.path.join(src_dir, fname)
    print(f"Converting {fname}...")

    # 用 pickle 直接加载（绕过 joblib 的 numpy 版本检查）
    import pickle

    with open(fpath, "rb") as f:
        # 用自定义 unpickler 重映射 numpy._core -> numpy.core
        class CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module.startswith("numpy._core"):
                    module = module.replace("numpy._core", "numpy.core")
                return super().find_class(module, name)


        data = CompatUnpickler(f).load()

    # 转换所有 numpy array 确保兼容性
    for k, v in data.items():
        if hasattr(v, "dtype"):
            data[k] = np.array(v)

    # 用当前 numpy 重新保存
    joblib.dump(data, fpath)
    print(f"  Done: {fname}")

print("All done!")
