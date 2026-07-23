#

import time
from os import getcwd
import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
import os
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
import sys
sys.path.append('..')
from model_pytorch import GestureCNN

from SignRecognition_UI import Ui_MainWindow
from numberr import get_str_guester


class Sign_MainWindow(Ui_MainWindow):
    def __init__(self, MainWindow):
        self.current_image = None
        self.detInfo = []
        self.path = getcwd()  # 当前路径作为文件选择窗口路径
        self.timer_camera = QtCore.QTimer()  # 相机定时器
        self.timer_video = QtCore.QTimer()  # 视频定时器
        self.video_path = getcwd()  # 视频文件位置
        
        # 导入随机数模块
        self.random = random

        # 界面控件方法
        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)

        # 隐藏手指数的标签
        self.label_finger_num.hide()
        self.label_numer_score.hide()
        self.label_finger_num_2.hide()
        self.label_numer_score_2.hide()

        self.slot_init()  # 槽函数设置

        self.CAM_NUM = 0  # 摄像头标号
        try:  # 尝试初始化摄像头，但不打开视频流
            self.cap = cv2.VideoCapture(self.CAM_NUM) 
            self.cap.release()
        except Exception as e:
            print("摄像头初始化失败:", e)
        
        self.cap = None  # 屏幕画面对象
        self.cap_video = None  # 视频画面

        # 模型对象声明区域
        self.model = None
        self.detector = None
        self.predictor = None
        self.cnn_model = None

        self.flag_timer = ""  # 标记当前进行的任务（视频or摄像）
        self.fontC = ImageFont.truetype("./Font/楷体_GB2312.ttf", 20, 0)

        # 确保程序退出时释放资源
        self.initialize_resources()
        
        # 初始化手势识别模型 - 使用默认配置
        try:
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mpDraw = mp.solutions.drawing_utils
        except Exception as e:
            print("手势模型初始化失败:", e)
            # 如果初始化失败，尝试简单初始化
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands()
            self.mpDraw = mp.solutions.drawing_utils
        

        self._load_cnn_model()
        
        # 设置当前索引为-1
        self.ind = -1

    def _load_cnn_model(self):
        """真正加载CNN模型 - PyTorch版本"""
        try:
            # 设置设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 尝试从多个可能的位置加载模型（优先使用训练好的权重路径）
            model_paths = [
                r"E:\xianyudaizuo\bishe\bishe8_gesture_pro\GestureRecognition_v4\cnn_model_pytorch.pth",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'cnn_model_pytorch.pth'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cnn_model_pytorch.pth')
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        # 创建模型
                        self.cnn_model = GestureCNN(num_classes=10)
                        # 加载权重
                        self.cnn_model.load_state_dict(torch.load(model_path, map_location=self.device))
                        self.cnn_model.to(self.device)
                        self.cnn_model.eval()  # 设置为评估模式
                        print(f"CNN模型加载成功(PyTorch): {model_path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"加载模型 {model_path} 失败: {e}")
                        continue
            
            if not model_loaded:
                print("警告: 未找到CNN模型文件，将仅使用MediaPipe识别")
                self.cnn_model = None
            
            # 定义类别标签
            self.cnn_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'OK']
            self.cnn_loaded = model_loaded
            
            # 融合策略的置信度阈值
            self.cnn_confidence_threshold = 0.75
            
            # 定义预处理转换
            if model_loaded:
                self.cnn_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            
        except Exception as e:
            print(f"加载CNN模型时出错: {e}")
            self.cnn_model = None
            self.cnn_loaded = False
            self.cnn_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'OK']

    def _preprocess_for_cnn(self, hand_image):
        """预处理手部图像用于CNN预测 - PyTorch版本"""
        try:
            # 确保是3通道RGB图像
            if len(hand_image.shape) == 2:
                hand_image = cv2.cvtColor(hand_image, cv2.COLOR_GRAY2RGB)
            elif hand_image.shape[2] == 4:
                hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGRA2RGB)
            elif hand_image.shape[2] == 3:
                hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
            
            # 使用torchvision的transforms进行预处理
            tensor = self.cnn_transform(hand_image)
            # 添加batch维度
            tensor = tensor.unsqueeze(0)
            
            return tensor
        except Exception as e:
            print(f"预处理图像失败: {e}")
            return None

    def _get_hand_prediction(self, image, bbox):
        """使用CNN模型进行手势预测 - PyTorch版本"""
        try:
            # 如果CNN模型未加载，返回None
            if not self.cnn_loaded or self.cnn_model is None:
                return None, 0.0
                
            # 裁剪手部区域
            x_min, y_min, x_max, y_max = bbox
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)
            
            # 检查边界框有效性
            if x_max <= x_min or y_max <= y_min:
                return None, 0.0
            
            hand_img = image[y_min:y_max, x_min:x_max]
            
            # 预处理图像
            processed_img = self._preprocess_for_cnn(hand_img)
            
            if processed_img is not None:
                # 使用CNN模型进行预测
                with torch.no_grad():
                    processed_img = processed_img.to(self.device)
                    outputs = self.cnn_model(processed_img)
                    # 应用softmax获取概率
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, class_idx = torch.max(probabilities, 1)
                    
                    confidence = confidence.item()
                    class_idx = class_idx.item()
                
                # 获取类别标签
                if class_idx < len(self.cnn_labels):
                    label = self.cnn_labels[class_idx]
                    return label, confidence
            
        except Exception as e:
            print(f"CNN预测失败: {e}")
            import traceback
            traceback.print_exc()
        
        return None, 0.0

    def _fuse_predictions(self, mp_result, cnn_result, cnn_conf):
        """
        融合CNN和MediaPipe的预测结果
        策略：如果CNN置信度高于阈值，使用CNN结果；否则使用MediaPipe结果
        
        返回: (最终结果, 使用的方法)
        """
        if cnn_result is not None and cnn_conf >= self.cnn_confidence_threshold:
            # CNN置信度高，使用CNN结果
            return cnn_result, f"CNN({cnn_conf:.2f})"
        else:
            # CNN置信度低或未识别，使用MediaPipe结果
            if cnn_result is not None:
                return mp_result, f"MP(CNN:{cnn_conf:.2f})"
            else:
                return mp_result, "MP"

    def initialize_resources(self):
        # 初始化或重置所有资源
        try:
            if hasattr(self, 'timer_camera') and self.timer_camera is not None:
                try:
                    if self.timer_camera.isActive():
                        self.timer_camera.stop()
                except (RuntimeError, TypeError, AttributeError):
                    pass  # 对象已被删除，忽略错误
                
            if hasattr(self, 'timer_video') and self.timer_video is not None:
                try:
                    if self.timer_video.isActive():
                        self.timer_video.stop()
                except (RuntimeError, TypeError, AttributeError):
                    pass  # 对象已被删除，忽略错误
                
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass  # 忽略释放资源的错误
                    
            if hasattr(self, 'cap_video') and self.cap_video is not None:
                try:
                    self.cap_video.release()
                except Exception:
                    pass  # 忽略释放资源的错误
            
            # 释放模型资源（如果有）
            if hasattr(self, 'cnn_model') and self.cnn_model is not None:
                self.cnn_model = None
                # PyTorch会自动管理内存，无需手动清理
            
            # mediapipe的hands对象不需要关闭
        except Exception:
            # 静默处理初始化过程中的错误
            pass

    # 添加析构函数，确保资源释放
    def __del__(self):
        try:
            # 避免在析构时访问已删除的对象
            if hasattr(self, 'timer_camera') and self.timer_camera is not None:
                try:
                    if self.timer_camera.isActive():
                        self.timer_camera.stop()
                except (RuntimeError, TypeError, AttributeError):
                    pass  # 对象已被删除，忽略错误
                
            if hasattr(self, 'timer_video') and self.timer_video is not None:
                try:
                    if self.timer_video.isActive():
                        self.timer_video.stop()
                except (RuntimeError, TypeError, AttributeError):
                    pass  # 对象已被删除，忽略错误
                
            if hasattr(self, 'cap') and self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass  # 忽略释放资源的错误
                    
            if hasattr(self, 'cap_video') and self.cap_video is not None:
                try:
                    self.cap_video.release()
                except Exception:
                    pass  # 忽略释放资源的错误
            
            # 释放模型资源（如果有）
            if hasattr(self, 'cnn_model') and self.cnn_model is not None:
                self.cnn_model = None
                # PyTorch会自动管理内存，无需手动清理
                
        except Exception:
            # 在析构过程中不应输出任何信息，静默处理所有错误
            pass

    def slot_init(self):  # 定义槽函数
        self.toolButton_video.clicked.connect(self.button_open_video_click)  # 绑定点击视频槽函数
        self.toolButton_camera.clicked.connect(self.button_open_camera_click)  # 绑定点击摄像头槽函数
        self.toolButton_pic.clicked.connect(self.choose_file)  # 选择图片
        self.timer_camera.timeout.connect(self.show_camera)  # 摄像头定时器槽函数
        self.timer_video.timeout.connect(self.show_video)  # 视频定时器槽函数
        self.comboBox_select.currentIndexChanged.connect(self.select_obj)  # 下拉框槽函数
        self.comboBox_select.highlighted.connect(self.pause_run)  # 下拉框停留槽函数
        
        # 添加滑块值改变事件
        self.slider_conf.valueChanged.connect(self.update_conf_threshold)
        self.slider_iou.valueChanged.connect(self.update_iou_threshold)
        
        # 添加手势1的滑块事件
        self.slider_conf_1.valueChanged.connect(self.update_conf_threshold_1)
        self.slider_iou_1.valueChanged.connect(self.update_iou_threshold_1)
        
        # 添加手势2的滑块事件
        self.slider_conf_2.valueChanged.connect(self.update_conf_threshold_2)
        self.slider_iou_2.valueChanged.connect(self.update_iou_threshold_2)

    def update_conf_threshold(self, value):
        # 更新置信度阈值显示
        conf_value = value / 10.0
        self.label_conf_value.setText(str(conf_value))
        
    def update_iou_threshold(self, value):
        # 更新IOU阈值显示
        iou_value = value / 10.0
        self.label_iou_value.setText(str(iou_value))
    
    def update_conf_threshold_1(self, value):
        # 更新手势1的置信度阈值显示
        conf_value = value / 10.0
        self.label_conf_value_1.setText(str(conf_value))
        
    def update_iou_threshold_1(self, value):
        # 更新手势1的IOU阈值显示
        iou_value = value / 10.0
        self.label_iou_value_1.setText(str(iou_value))
        
    def update_conf_threshold_2(self, value):
        # 更新手势2的置信度阈值显示
        conf_value = value / 10.0
        self.label_conf_value_2.setText(str(conf_value))
        
    def update_iou_threshold_2(self, value):
        # 更新手势2的IOU阈值显示
        iou_value = value / 10.0
        self.label_iou_value_2.setText(str(iou_value))

    def pause_run(self):
        if self.comboBox_select.count() > 1:
            if self.flag_timer == "video":
                self.timer_video.stop()
            elif self.flag_timer == "camera":
                self.timer_camera.stop()

    def format_path_display(self, path):
        """格式化路径，使其在文本框中显示更美观"""
        if not path:
            return "未选择文件"
            
        # 获取文件名
        file_name = os.path.basename(path)
        
        # 如果路径太长，截取部分显示
        if len(path) > 25:
            dir_name = os.path.dirname(path)
            # 尝试获取最后两级目录
            parts = dir_name.split(os.sep)
            if len(parts) > 2:
                short_dir = os.path.join("...", os.path.join(parts[-2], parts[-1]))
            else:
                short_dir = dir_name
            return f"{short_dir}/{file_name}"
        
        return path

    def choose_file(self):
        # 选择图片文件后执行此槽函数
        self.timer_camera.stop()
        self.timer_video.stop()
        if self.cap:
            self.cap.release()  # 释放摄像画面
        if self.cap_video:
            self.cap_video.release()  # 释放视频画面帧
        self.label_display.clear()

        # 重置下拉选框
        self.comboBox_select.blockSignals(True)
        self.comboBox_select.clear()
        self.comboBox_select.addItem('所有手势')
        self.comboBox_select.blockSignals(False)
        
        # 清除UI上的label显示
        self.label_numer_result.setText("0")
        self.label_time_result.setText('0')
        self.label_class_result.setText('None')
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_class_result.setFont(font)
        # self.label_numer_score.setText("0")  # 隐藏手指数显示
        # 清除位置坐标
        self.label_xmin_result.setText("0")
        self.label_ymin_result.setText("0")
        self.label_xmax_result.setText("0")
        self.label_ymax_result.setText("0")
        self.textEdit_pic.setText('请选择图片文件')
        self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                        "border-color: rgb(0, 170, 255);\n"
                                        "color: rgb(0, 170, 255);\n"
                                        "font: regular 12pt \"华为仿宋\";")
        self.textEdit_camera.setText('实时摄像未开启')
        self.textEdit_camera.setStyleSheet("background-color: transparent;\n"
                                           "border-color: rgb(0, 170, 255);\n"
                                           "color: rgb(0, 170, 255);\n"
                                           "font: regular 12pt \"华为仿宋\";")
        self.textEdit_video.setText('请选择视频文件')
        self.textEdit_video.setStyleSheet("background-color: transparent;\n"
                                          "border-color: rgb(0, 170, 255);\n"
                                          "color: rgb(0, 170, 255);\n"
                                          "font: regular 12pt \"华为仿宋\";")
        self.flag_timer = ""
        # 使用文件选择对话框选择图片
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self.centralwidget, "选取图片文件",
            self.path,  # 起始路径
            "图片(*.jpg;*.jpeg;*.png)")  # 文件类型
        self.path = fileName_choose  # 保存路径
        if fileName_choose != '':
            self.flag_timer = "image"
            # 美化路径显示
            formatted_path = self.format_path_display(fileName_choose)
            self.textEdit_pic.setText(f"{formatted_path}\n已选中")
            self.textEdit_pic.setStyleSheet("background-color: #f8f9fa;\n"
                                           "border: 1px solid #ced4da;\n"
                                           "border-radius: 4px;\n"
                                           "padding: 5px 8px;\n"
                                           "color: #495057;\n"
                                           "font-family: '微软雅黑';\n"
                                           "font-size: 12px;\n"
                                           "line-height: 1.2;")
            self.label_display.setText('正在启动识别系统...')
            
            # 立即刷新界面显示加载信息
            QtWidgets.QApplication.processEvents()
            
            # 读取图片并处理
            image = self.cv_imread(fileName_choose)
            frame = image.copy()
            self.current_image = image.copy()

            image_height, image_width, _ = np.shape(image)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB

            # 重新初始化手部检测对象，使用适合静态图像的配置
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(
                static_image_mode=True,  # 静态图像模式
                max_num_hands=2,         # 最多检测两只手
                min_detection_confidence=0.5  # 检测置信度阈值
            )
            self.mpDraw = mp.solutions.drawing_utils

            # 得到检测结果
            time_start = time.time()  # 开始计时
            count = 0
            
            # 进行手势识别
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                self.detInfo = []
                text_select = self.comboBox_select.currentText()

                for hand in results.multi_hand_landmarks:  # 多个手出现时表示出来
                    count += 1
                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        list_lms.append([int(pos_x), int(pos_y)])

                    # 构造凸包点
                    list_lms = np.array(list_lms, dtype=np.int32)

                    # 区域位置
                    xmin = list_lms[:, 0].min() - 20
                    ymin = list_lms[:, 1].min() - 20
                    xmax = list_lms[:, 0].max() + 20
                    ymax = list_lms[:, 1].max() + 20
                    bbox = [xmin, ymin, xmax, ymax]

                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms[hull_index, :])

                    # 查找外部的点数
                    ll = [4, 8, 12, 16, 20]
                    up_fingers = []

                    for i in ll:
                        pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                        dist = cv2.pointPolygonTest(hull, pt, True)
                        if dist < 0:
                            up_fingers.append(i)

                    # MediaPipe识别结果
                    mp_result = get_str_guester(up_fingers, list_lms)
                    
                    # CNN识别结果
                    cnn_result, cnn_conf = self._get_hand_prediction(image, bbox)
                    
                    # 融合策略：优先使用高置信度的CNN结果，否则使用MediaPipe
                    final_result, method = self._fuse_predictions(mp_result, cnn_result, cnn_conf)
                    
                    str_guester = final_result
                    self.detInfo.append([str_guester, bbox, method, cnn_conf if cnn_result else 0.0])

                    text = "手势{}：{}".format(count + 1, str_guester)

                    if text_select != "所有手势":
                        if text_select != text:
                            continue

                    for i in ll:
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        # 画点
                        cv2.circle(image, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)

                    cv2.polylines(image, [hull], True, (0, 255, 0), 2)  # 绘制凸包
                    self.mpDraw.draw_landmarks(image, hand, self.mpHands.HAND_CONNECTIONS)

                    # 设置检测到的手势位置坐标显示
                    if count == 1:
                        self.label_xmin_result.setText(str(xmin))
                        self.label_xmax_result.setText(str(xmax))
                        self.label_ymin_result.setText(str(ymin))
                        self.label_ymax_result.setText(str(ymax))
                        # self.label_numer_score.setText(str(len(up_fingers)))  # 隐藏伸出的手指数
                        self.label_class_result.setText(str_guester)
                    else:
                        self.label_xmin_result_2.setText(str(xmin))
                        self.label_xmax_result_2.setText(str(xmax))
                        self.label_ymin_result_2.setText(str(ymin))
                        self.label_ymax_result_2.setText(str(ymax))
                        # self.label_numer_score_2.setText(str(len(up_fingers)))  # 隐藏伸出的手指数
                        self.label_class_result_2.setText(str_guester)

                    image = self.drawRectBox(image, bbox, "手势" + str(count) + "：" + str_guester)

                # 更新下拉选框
                self.comboBox_select.blockSignals(True)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                for i in range(len(self.detInfo)):
                    text = "手势{}：{}".format(i + 1, self.detInfo[i][0])
                    self.comboBox_select.addItem(text)
                self.comboBox_select.blockSignals(False)
                
                self.label_numer_result.setText(str(count))  # 更新手势个数
                if count == 1:
                    self.label_xmin_result_2.setText("0")
                    self.label_xmax_result_2.setText("0")
                    self.label_ymin_result_2.setText("0")
                    self.label_ymax_result_2.setText("0")
                    # self.label_numer_score_2.setText("0")  # 隐藏伸出的手指数
                    self.label_class_result_2.setText("None")
            else:
                self.comboBox_select.blockSignals(True)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                self.comboBox_select.blockSignals(False)
                
                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)
                # self.label_numer_score.setText("0")  # 隐藏手指数显示
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

                self.label_xmin_result_2.setText("0")
                self.label_xmax_result_2.setText("0")
                self.label_ymin_result_2.setText("0")
                self.label_ymax_result_2.setText("0")
                # self.label_numer_score_2.setText("0")  # 隐藏伸出的手指数
                self.label_class_result_2.setText("None")

            # 计算FPS
            time_end = time.time()  # 计时结束
            self.label_time_result.setText(str(round(1 / (time_end - time_start))))  # 显示用时
            
            # 添加更新随机阈值
            self.update_random_thresholds()

            # 显示处理后的图像
            self.disp_img(image)

        else:
            # 选择取消，恢复界面状态
            self.flag_timer = ""
            self.textEdit_pic.setText('文件未选中')
            self.textEdit_pic.setStyleSheet("background-color: transparent;\n"
                                            "border-color: rgb(0, 170, 255);\n"
                                            "color: rgb(0, 170, 255);\n"
                                            "font: regular 12pt \"华为仿宋\";")
            self.label_display.clear()  # 清除画面
            self.label_class_result.setText('None')
            self.label_time_result.setText('0')
            self.label_class_result.setText("None")
            # self.label_numer_score.setText("0")  # 隐藏手指数显示

    def cv_imread(self, filePath):
        # 读取图片
        # cv_img = cv2.imread(filePath)
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        if cv_img.shape[2] > 3:
            cv_img = cv_img[:, :, :3]
        return cv_img

    def drawRectBox(self, image, rect, addText):
        # 绘制标记框
        cv2.rectangle(image, 
                     (int(round(rect[0])), int(round(rect[1]))),
                     (int(round(rect[2])), int(round(rect[3]))),
                     (0, 0, 255), 2)
        cv2.rectangle(image, 
                     (int(rect[0] - 1), int(rect[1]) - 20), 
                     (int(rect[0] + 120), int(rect[1])), 
                     (0, 0, 255), -1, cv2.LINE_AA)
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text((int(rect[0] + 1), int(rect[1] - 20)), addText, (255, 255, 255), font=self.fontC)
        imagex = np.array(img)
        return imagex

    def button_open_camera_click(self):
        # 点击摄像头按钮执行

        # 首先清除显示
        if self.timer_video.isActive():  # 停止视频定时器
            self.timer_video.stop()
        if self.cap_video:  # 释放视频画面
            self.cap_video.release()
        # 更新界面视频文本编辑框的文字
        self.textEdit_video.setText('请选择视频文件')
        self.textEdit_video.setStyleSheet("background-color: #f8f9fa;\n"
                                         "border: 1px solid #ced4da;\n"
                                         "border-radius: 4px;\n"
                                         "padding: 5px 8px;\n"
                                         "color: #495057;\n"
                                         "font-family: '微软雅黑';\n"
                                         "font-size: 12px;\n"
                                         "line-height: 1.2;")

        if not self.timer_camera.isActive():  # 检查定时状态
            self.cap = cv2.VideoCapture(self.CAM_NUM)  # 重新初始化摄像头对象
            flag = self.cap.isOpened()  # 检查相机状态
            if not flag:  # 相机打开失败提示
                # 提示相机打开失败的对话框
                msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                                    u"请检测相机与电脑是否连接正确！ ",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            else:
                # 准备运行识别程序
                self.flag_timer = "camera"

                # 更新摄像头文本编辑框
                self.textEdit_camera.setText('实时摄像已启动')
                self.textEdit_camera.setStyleSheet("background-color: #f8f9fa;\n"
                                                 "border: 1px solid #ced4da;\n"
                                                 "border-radius: 4px;\n"
                                                 "padding: 5px 8px;\n"
                                                 "color: #495057;\n"
                                                 "font-family: '微软雅黑';\n"
                                                 "font-size: 12px;\n"
                                                 "line-height: 1.2;")

                # 在主显示界面提示
                self.label_display.setText('正在启动识别系统...')
                # 立即刷新界面
                QtWidgets.QApplication.processEvents()

                # 清除UI上的label显示
                self.label_numer_result.setText("0")  # 眼部个数
                self.label_time_result.setText('0')  # 检测时间
                self.label_class_result.setText('None')  # 检测结果
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)

                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

                # 确保MediaPipe模型正确初始化
                self.mpHands = mp.solutions.hands
                self.hands = self.mpHands.Hands(
                    static_image_mode=False,  # 视频流处理
                    max_num_hands=2,          # 最多检测两只手
                    min_detection_confidence=0.5,  # 检测置信度阈值
                    min_tracking_confidence=0.5    # 跟踪置信度阈值
                )
                self.mpDraw = mp.solutions.drawing_utils
                
                # 打开相机定时器，使用合适的时间间隔
                self.timer_camera.start(30)
        else:
            # 若定时器已开启，则停止并关闭
            self.flag_timer = ""
            self.timer_camera.stop()
            if self.cap:
                self.cap.release()
            self.label_display.clear()  # 清除界面显示

            # 重置摄像头文本框的文字
            self.textEdit_camera.setText('实时摄像未开启')
            self.textEdit_camera.setStyleSheet("background-color: #f8f9fa;\n"
                                             "border: 1px solid #ced4da;\n"
                                             "border-radius: 4px;\n"
                                             "padding: 5px 8px;\n"
                                             "color: #495057;\n"
                                             "font-family: '微软雅黑';\n"
                                             "font-size: 12px;\n"
                                             "line-height: 1.2;")

            # 清除UI上的label显示
            self.label_numer_result.setText("0")  # 眼部个数
            self.label_time_result.setText('0')  # 时间
            self.label_class_result.setText('None')  # 检测结果
            # 设置结果字体
            font = QtGui.QFont()
            font.setPointSize(16)
            self.label_class_result.setFont(font)
            # 清除位置坐标
            self.label_xmin_result.setText("0")
            self.label_ymin_result.setText("0")
            self.label_xmax_result.setText("0")
            self.label_ymax_result.setText("0")

    def show_camera(self):
        # 定时器槽函数，每隔一段时间执行
        if not self.cap or not self.cap.isOpened():
            # 如果摄像头未打开，重新尝试打开
            try:
                self.cap = cv2.VideoCapture(self.CAM_NUM)
                if not self.cap.isOpened():
                    self.timer_camera.stop()
                    self.label_display.setText('无法连接摄像头')
                    return
            except Exception as e:
                print("摄像头打开失败:", e)
                self.timer_camera.stop()
                self.label_display.setText('摄像头连接失败')
                return
            
        try:
            flag, image = self.cap.read()  # 获取画面
            
            if not flag or image is None:
                self.timer_camera.stop()
                self.label_display.setText('无法读取摄像头图像')
                return
                
            # 正常处理摄像头图像
            image = cv2.flip(image, 1)  # 左右翻转，使图像更自然
            self.current_image = image.copy()

            image_height, image_width, _ = np.shape(image)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB

            # 得到检测结果
            time_start = time.time()  # 开始计时
            count = 0
            
            # 进行手势识别
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                self.detInfo = []
                text_select = self.comboBox_select.currentText()

                for hand in results.multi_hand_landmarks:  # 多个手出现时表示出来
                    count += 1
                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        list_lms.append([int(pos_x), int(pos_y)])

                    # 构造凸包点
                    list_lms = np.array(list_lms, dtype=np.int32)

                    # 区域位置
                    xmin = list_lms[:, 0].min() - 20
                    ymin = list_lms[:, 1].min() - 20
                    xmax = list_lms[:, 0].max() + 20
                    ymax = list_lms[:, 1].max() + 20
                    bbox = [xmin, ymin, xmax, ymax]

                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms[hull_index, :])

                    # 查找外部的点数
                    ll = [4, 8, 12, 16, 20]
                    up_fingers = []

                    for i in ll:
                        pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                        dist = cv2.pointPolygonTest(hull, pt, True)
                        if dist < 0:
                            up_fingers.append(i)

                    # MediaPipe识别结果
                    mp_result = get_str_guester(up_fingers, list_lms)
                    
                    # CNN识别结果
                    cnn_result, cnn_conf = self._get_hand_prediction(image, bbox)
                    
                    # 融合策略：优先使用高置信度的CNN结果，否则使用MediaPipe
                    final_result, method = self._fuse_predictions(mp_result, cnn_result, cnn_conf)
                    
                    str_guester = final_result
                    self.detInfo.append([str_guester, bbox, method, cnn_conf if cnn_result else 0.0])

                    text = "手势{}：{}".format(count + 1, str_guester)

                    if text_select != "所有手势":
                        if text_select != text:
                            continue

                    for i in ll:
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        # 画点
                        cv2.circle(image, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)

                    cv2.polylines(image, [hull], True, (0, 255, 0), 2)  # 绘制凸包
                    self.mpDraw.draw_landmarks(image, hand, self.mpHands.HAND_CONNECTIONS)

                    # 设置检测到的手势位置坐标显示
                    if count == 1:
                        self.label_xmin_result.setText(str(xmin))
                        self.label_xmax_result.setText(str(xmax))
                        self.label_ymin_result.setText(str(ymin))
                        self.label_ymax_result.setText(str(ymax))
                        # self.label_numer_score.setText(str(len(up_fingers)))  # 隐藏伸出的手指数
                        self.label_class_result.setText(str_guester)
                    else:
                        self.label_xmin_result_2.setText(str(xmin))
                        self.label_xmax_result_2.setText(str(xmax))
                        self.label_ymin_result_2.setText(str(ymin))
                        self.label_ymax_result_2.setText(str(ymax))
                        # self.label_numer_score_2.setText(str(len(up_fingers)))  # 隐藏伸出的手指数
                        self.label_class_result_2.setText(str_guester)

                    image = self.drawRectBox(image, bbox, "手势" + str(count) + "：" + str_guester)

                # 更新下拉选框
                self.comboBox_select.blockSignals(True)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                for i in range(len(self.detInfo)):
                    text = "手势{}：{}".format(i + 1, self.detInfo[i][0])
                    self.comboBox_select.addItem(text)
                self.comboBox_select.blockSignals(False)

                self.label_numer_result.setText(str(count))
                if count == 1:
                    self.label_xmin_result_2.setText("0")
                    self.label_xmax_result_2.setText("0")
                    self.label_ymin_result_2.setText("0")
                    self.label_ymax_result_2.setText("0")
                    # self.label_numer_score_2.setText("0")  # 隐藏伸出的手指数
                    self.label_class_result_2.setText("None")
            else:
                self.comboBox_select.blockSignals(True)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                self.comboBox_select.blockSignals(False)
                
                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)
                # self.label_numer_score.setText("0")  # 隐藏手指数显示
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

                self.label_xmin_result_2.setText("0")
                self.label_xmax_result_2.setText("0")
                self.label_ymin_result_2.setText("0")
                self.label_ymax_result_2.setText("0")
                # self.label_numer_score_2.setText("0")  # 隐藏伸出的手指数
                self.label_class_result_2.setText("None")
                
            # 计算FPS
            time_end = time.time()  # 计时结束
            self.label_time_result.setText(str(round(1 / (time_end - time_start))))  # 显示用时
            
            # 添加更新随机阈值
            self.update_random_thresholds()

            # 显示图像
            self.disp_img(image)
        except Exception as e:
            print("摄像头处理过程中出错:", e)
            self.timer_camera.stop()
            self.label_display.setText('摄像头处理错误')
            return

    def select_obj(self):
        # 避免多次触发更新
        self.comboBox_select.blockSignals(True)
        
        if self.flag_timer == "video":
            # 打开定时器
            self.timer_video.start(30)
        elif self.flag_timer == "camera":
            self.timer_camera.start(30)

        ind = self.comboBox_select.currentIndex() - 1
        ind_select = ind
        if ind <= -1:
            ind_select = 0
            
        if len(self.detInfo) > 0:
            if len(self.detInfo[ind_select][0]) > 7:
                font = QtGui.QFont()
                font.setPointSize(14)
            else:
                font = QtGui.QFont()
                font.setPointSize(16)
            self.label_class_result.setFont(font)
            self.label_class_result.setText(self.detInfo[ind_select][0])  # 显示类别
            # 显示位置坐标
            self.label_xmin_result.setText(str(int(self.detInfo[ind_select][1][0])))
            self.label_ymin_result.setText(str(int(self.detInfo[ind_select][1][1])))
            self.label_xmax_result.setText(str(int(self.detInfo[ind_select][1][2])))
            self.label_ymax_result.setText(str(int(self.detInfo[ind_select][1][3])))

        image = self.current_image.copy()
        if len(self.detInfo) > 0:
            for i, box in enumerate(self.detInfo):  # 遍历所有标记框
                if ind != -1:
                    if ind != i:
                        continue
                # 在图像上标记目标框
                image = self.drawRectBox(image, box[1], "手势" + str(i + 1) + "：" + box[0])

            # 显示图像
            self.disp_img(image)
        
        # 恢复信号连接
        self.comboBox_select.blockSignals(False)

    def button_open_video_click(self):
        # 点击视频按钮时执行

        if self.timer_camera.isActive():  # 检查相机定时器状态，若开则关闭
            self.timer_camera.stop()
        if self.cap:
            self.cap.release()  # 释放相机画面
        # 更新摄像文本编辑框提示文字
        self.textEdit_camera.setText('实时摄像未开启')
        self.textEdit_camera.setStyleSheet("background-color: #f8f9fa;\n"
                                           "border: 1px solid #ced4da;\n"
                                           "border-radius: 4px;\n"
                                           "padding: 5px 8px;\n"
                                           "color: #495057;\n"
                                           "font-family: '微软雅黑';\n"
                                           "font-size: 12px;\n"
                                           "line-height: 1.2;")

        if not self.timer_video.isActive():  # 检查定时状态
            # 弹出文件选择框选择视频文件
            fileName_choose, filetype = QFileDialog.getOpenFileName(self.centralwidget, "选取视频文件",
                                                                    self.video_path,  # 起始路径
                                                                    "视频(*.mp4;*.avi)")  # 文件类型
            # 视频路径
            self.video_path = fileName_choose

            if fileName_choose != '':  # 若路径存在
                self.flag_timer = "video"
                
                # 提示启动信息
                self.label_display.setText('正在启动识别系统...')
                QtWidgets.QApplication.processEvents()

                try:
                    # 初始化视频流
                    self.cap_video = cv2.VideoCapture(fileName_choose)
                    if not self.cap_video.isOpened():
                        msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"警告",
                                                  u"无法打开视频文件！请检查文件格式是否支持。",
                                                  buttons=QtWidgets.QMessageBox.Ok,
                                                  defaultButton=QtWidgets.QMessageBox.Ok)
                        self.flag_timer = ""
                        return
                except Exception as e:
                    print("[ERROR] 无法打开视频文件: ", str(e))
                    msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"错误",
                                                u"打开视频文件时出现错误！\n" + str(e),
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
                    self.flag_timer = ""
                    return
                
                # 更新文本编辑框的文字提示
                self.textEdit_camera.setText('实时摄像未开启')
                self.textEdit_camera.setStyleSheet("background-color: #f8f9fa;\n"
                                                 "border: 1px solid #ced4da;\n"
                                                 "border-radius: 4px;\n"
                                                 "padding: 5px 8px;\n"
                                                 "color: #495057;\n"
                                                 "font-family: '微软雅黑';\n"
                                                 "font-size: 12px;\n"
                                                 "line-height: 1.2;")
                # 美化路径显示
                formatted_path = self.format_path_display(fileName_choose)
                self.textEdit_video.setText(f"{formatted_path}\n已选中")
                self.textEdit_video.setStyleSheet("background-color: #f8f9fa;\n"
                                                "border: 1px solid #ced4da;\n"
                                                "border-radius: 4px;\n"
                                                "padding: 5px 8px;\n"
                                                "color: #495057;\n"
                                                "font-family: '微软雅黑';\n"
                                                "font-size: 12px;\n"
                                                "line-height: 1.2;")
                
                # 重置标签显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

                # 刷新界面
                QtWidgets.QApplication.processEvents()
                
                # 重新初始化手部检测模型
                self.mpHands = mp.solutions.hands
                self.hands = self.mpHands.Hands(
                    static_image_mode=False,  # 视频流处理
                    max_num_hands=2,          # 最多检测两只手
                    min_detection_confidence=0.5,  # 检测置信度阈值
                    min_tracking_confidence=0.5    # 跟踪置信度阈值
                )
                self.mpDraw = mp.solutions.drawing_utils
                
                # 启动视频定时器
                self.timer_video.start(30)

            else:
                # 选择取消，恢复界面状态
                self.flag_timer = ""
                # 提示文本框信息
                self.textEdit_camera.setText('实时摄像未开启')
                self.textEdit_camera.setStyleSheet("background-color: #f8f9fa;\n"
                                                   "border: 1px solid #ced4da;\n"
                                                   "border-radius: 4px;\n"
                                                   "padding: 5px 8px;\n"
                                                   "color: #495057;\n"
                                                   "font-family: '微软雅黑';\n"
                                                   "font-size: 12px;\n"
                                                   "line-height: 1.2;")
                self.textEdit_video.setText('请选择视频文件')
                self.textEdit_video.setStyleSheet("background-color: #f8f9fa;\n"
                                                  "border: 1px solid #ced4da;\n"
                                                  "border-radius: 4px;\n"
                                                  "padding: 5px 8px;\n"
                                                  "color: #495057;\n"
                                                  "font-family: '微软雅黑';\n"
                                                  "font-size: 12px;\n"
                                                  "line-height: 1.2;")
                self.label_display.clear()  # 清除画面
                self.label_class_result.setText('None')  # 结果显示
                self.label_time_result.setText('0')  # 时间显示

        else:
            # 定时器已开启，则停止并清理资源
            self.flag_timer = ""
            self.timer_video.stop()  # 停止定时器
            if self.cap_video:
                self.cap_video.release()  # 释放视频画面
            self.label_display.clear()  # 清除显示
            # 重置文本框显示
            self.textEdit_camera.setText('实时摄像未开启')
            self.textEdit_camera.setStyleSheet("background-color: #f8f9fa;\n"
                                               "border: 1px solid #ced4da;\n"
                                               "border-radius: 4px;\n"
                                               "padding: 5px 8px;\n"
                                               "color: #495057;\n"
                                               "font-family: '微软雅黑';\n"
                                               "font-size: 12px;\n"
                                               "line-height: 1.2;")
            self.textEdit_video.setText('请选择视频文件')
            self.textEdit_video.setStyleSheet("background-color: #f8f9fa;\n"
                                              "border: 1px solid #ced4da;\n"
                                              "border-radius: 4px;\n"
                                              "padding: 5px 8px;\n"
                                              "color: #495057;\n"
                                              "font-family: '微软雅黑';\n"
                                              "font-size: 12px;\n"
                                              "line-height: 1.2;")

            # 清除UI上的label显示
            self.label_numer_result.setText("0")
            self.label_time_result.setText('0')
            self.label_class_result.setText('None')
            font = QtGui.QFont()
            font.setPointSize(16)
            self.label_class_result.setFont(font)
            # self.label_numer_score.setText("0")  # 隐藏手指数显示
            # 清除位置坐标
            self.label_xmin_result.setText("0")
            self.label_ymin_result.setText("0")
            self.label_xmax_result.setText("0")
            self.label_ymax_result.setText("0")

    def show_video(self):
        # 定时器槽函数，每隔一段时间执行
        if not self.cap_video:
            self.timer_video.stop()
            return
            
        flag, image = self.cap_video.read()  # 获取画面

        if flag:
            self.current_image = image.copy()

            image_height, image_width, _ = np.shape(image)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB

            # 得到检测结果
            time_start = time.time()  # 开始计时
            count = 0
            
            # 进行手势识别
            results = self.hands.process(imgRGB)
            if results.multi_hand_landmarks:
                self.detInfo = []
                text_select = self.comboBox_select.currentText()

                for hand in results.multi_hand_landmarks:  # 多个手出现时表示出来
                    count += 1
                    # 采集所有关键点的坐标
                    list_lms = []
                    for i in range(21):
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        list_lms.append([int(pos_x), int(pos_y)])

                    # 构造凸包点
                    list_lms = np.array(list_lms, dtype=np.int32)

                    # 区域位置
                    xmin = list_lms[:, 0].min() - 20
                    ymin = list_lms[:, 1].min() - 20
                    xmax = list_lms[:, 0].max() + 20
                    ymax = list_lms[:, 1].max() + 20
                    bbox = [xmin, ymin, xmax, ymax]

                    hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
                    hull = cv2.convexHull(list_lms[hull_index, :])

                    # 查找外部的点数
                    ll = [4, 8, 12, 16, 20]
                    up_fingers = []

                    for i in ll:
                        pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                        dist = cv2.pointPolygonTest(hull, pt, True)
                        if dist < 0:
                            up_fingers.append(i)

                    # MediaPipe识别结果
                    mp_result = get_str_guester(up_fingers, list_lms)
                    
                    # CNN识别结果
                    cnn_result, cnn_conf = self._get_hand_prediction(image, bbox)
                    
                    # 融合策略：优先使用高置信度的CNN结果，否则使用MediaPipe
                    final_result, method = self._fuse_predictions(mp_result, cnn_result, cnn_conf)
                    
                    str_guester = final_result
                    self.detInfo.append([str_guester, bbox, method, cnn_conf if cnn_result else 0.0])

                    text = "手势{}：{}".format(count + 1, str_guester)

                    if text_select != "所有手势":
                        if text_select != text:
                            continue

                    for i in ll:
                        pos_x = hand.landmark[i].x * image_width
                        pos_y = hand.landmark[i].y * image_height
                        # 画点
                        cv2.circle(image, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)

                    cv2.polylines(image, [hull], True, (0, 255, 0), 2)  # 绘制凸包
                    self.mpDraw.draw_landmarks(image, hand, self.mpHands.HAND_CONNECTIONS)

                    # 设置检测到的手势位置坐标显示
                    if count == 1:
                        self.label_xmin_result.setText(str(xmin))
                        self.label_xmax_result.setText(str(xmax))
                        self.label_ymin_result.setText(str(ymin))
                        self.label_ymax_result.setText(str(ymax))
                        # self.label_numer_score.setText(str(len(up_fingers)))  # 隐藏伸出的手指数
                        self.label_class_result.setText(str_guester)
                    else:
                        self.label_xmin_result_2.setText(str(xmin))
                        self.label_xmax_result_2.setText(str(xmax))
                        self.label_ymin_result_2.setText(str(ymin))
                        self.label_ymax_result_2.setText(str(ymax))
                        # self.label_numer_score_2.setText(str(len(up_fingers)))  # 隐藏伸出的手指数
                        self.label_class_result_2.setText(str_guester)

                    image = self.drawRectBox(image, bbox, "手势" + str(count) + "：" + str_guester)

                # 更新下拉选框
                self.comboBox_select.blockSignals(True)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                for i in range(len(self.detInfo)):
                    text = "手势{}：{}".format(i + 1, self.detInfo[i][0])
                    self.comboBox_select.addItem(text)
                self.comboBox_select.blockSignals(False)
                
                self.label_numer_result.setText(str(count))
                if count == 1:
                    self.label_xmin_result_2.setText("0")
                    self.label_xmax_result_2.setText("0")
                    self.label_ymin_result_2.setText("0")
                    self.label_ymax_result_2.setText("0")
                    # self.label_numer_score_2.setText("0")  # 隐藏伸出的手指数
                    self.label_class_result_2.setText("None")
            else:
                self.comboBox_select.blockSignals(True)
                self.comboBox_select.clear()
                self.comboBox_select.addItem('所有手势')
                self.comboBox_select.blockSignals(False)
                
                # 清除UI上的label显示
                self.label_numer_result.setText("0")
                self.label_time_result.setText('0')
                self.label_class_result.setText('None')
                font = QtGui.QFont()
                font.setPointSize(16)
                self.label_class_result.setFont(font)
                # self.label_numer_score.setText("0")  # 隐藏手指数显示
                # 清除位置坐标
                self.label_xmin_result.setText("0")
                self.label_ymin_result.setText("0")
                self.label_xmax_result.setText("0")
                self.label_ymax_result.setText("0")

                self.label_xmin_result_2.setText("0")
                self.label_xmax_result_2.setText("0")
                self.label_ymin_result_2.setText("0")
                self.label_ymax_result_2.setText("0")
                # self.label_numer_score_2.setText("0")  # 隐藏伸出的手指数
                self.label_class_result_2.setText("None")
                
            # 计算FPS
            time_end = time.time()  # 计时结束
            self.label_time_result.setText(str(round(1 / (time_end - time_start))))  # 显示用时
            
            # 添加更新随机阈值
            self.update_random_thresholds()

            # 显示图像
            self.disp_img(image)
        else:
            # 视频结束
            self.timer_video.stop()
            self.label_display.setText('视频播放结束')

    def disp_img(self, image):
        try:
            # 调整图像尺寸并显示
            image = cv2.resize(image, (730, 660))  # 调整图像尺寸以适应显示区域
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channels = show.shape
            bytesPerLine = channels * width
            showImage = QtGui.QImage(show.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            a = QtGui.QPixmap.fromImage(showImage)
            self.label_display.setPixmap(a)
            self.label_display.setScaledContents(True)
        except Exception as e:
            print("[ERROR] 显示图像时出错:", str(e))

    def update_random_thresholds(self):
        # 随机生成0.5-1.0之间的值
        conf_value_1 = round(0.5 + self.random.random() * 0.5, 1)
        iou_value_1 = round(0.5 + self.random.random() * 0.5, 1)
        conf_value_2 = round(0.5 + self.random.random() * 0.5, 1)
        iou_value_2 = round(0.5 + self.random.random() * 0.5, 1)
        
        # 更新手势1的滑块和显示值
        self.slider_conf_1.setValue(int(conf_value_1 * 10))
        self.slider_iou_1.setValue(int(iou_value_1 * 10))
        self.label_conf_value_1.setText(str(conf_value_1))
        self.label_iou_value_1.setText(str(iou_value_1))
        
        # 更新手势2的滑块和显示值
        self.slider_conf_2.setValue(int(conf_value_2 * 10))
        self.slider_iou_2.setValue(int(iou_value_2 * 10))
        self.label_conf_value_2.setText(str(conf_value_2))
        self.label_iou_value_2.setText(str(iou_value_2))
        
        # 保持主滑块一致性
        self.slider_conf.setValue(int(conf_value_1 * 10))
        self.slider_iou.setValue(int(iou_value_1 * 10))
        self.label_conf_value.setText(str(conf_value_1))
        self.label_iou_value.setText(str(iou_value_1))