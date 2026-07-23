import os
import warnings

from SignRecognition import Sign_MainWindow
from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QMainWindow
import atexit

def ensure_model_dirs():
    """确保模型目录存在"""
    # 创建模型存储目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    model_path = os.path.join(model_dir, 'cnn_model.h5')
    if os.path.exists(model_path):
        pass
    
    return True

def cleanup():
    # 程序退出时执行的清理函数
    try:
        # 释放全局应用资源
        app = QApplication.instance()
        if app:
            app.closeAllWindows()
            app.processEvents()
    except Exception:
        # 静默处理清理过程中的错误
        pass

if __name__ == '__main__':
    # 忽略警告
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings(action='ignore')
    
    # 确保模型目录存在
    ensure_model_dirs()
    
    app = QApplication(argv)
    
    # 注册退出清理函数
    atexit.register(cleanup)

    window = QMainWindow()
    window.setWindowTitle("智能手势识别系统 v2.0")
    # 设置窗口样式为渐变背景
    window.setStyleSheet("""
        QMainWindow {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, 
                                 stop:0 #f8f9fa, stop:1 #e9ecef);
        }
    """)
    
    ui = Sign_MainWindow(window)

    window.show()
    result = app.exec_()
    
    # 在退出前确保释放资源
    if hasattr(ui, 'initialize_resources'):
        try:
            # 先确保所有定时器停止
            if hasattr(ui, 'timer_camera'):
                try:
                    if ui.timer_camera.isActive():
                        ui.timer_camera.stop()
                except Exception:
                    pass
                    
            if hasattr(ui, 'timer_video'):
                try:
                    if ui.timer_video.isActive():
                        ui.timer_video.stop()
                except Exception:
                    pass
            
            # 然后初始化/释放其他资源        
            ui.initialize_resources()
        except Exception:
            # 静默处理任何异常
            pass
    
    # 确保所有事件处理完成
    app.processEvents()
    
    exit(result)
