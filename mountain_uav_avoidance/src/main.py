import sys
import traceback
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def excepthook(exc_type, exc_value, exc_traceback):
    """全局异常捕获"""
    print("未捕获的异常:")
    traceback.print_exception(exc_type, exc_value, exc_traceback)

def main():
    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
