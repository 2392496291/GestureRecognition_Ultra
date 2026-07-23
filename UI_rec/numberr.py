import mediapipe as mp
import cv2
import numpy as np


def get_angle(v1, v2):
    angle = np.dot(v1, v2) / (np.sqrt(np.sum(v1 * v1)) * np.sqrt(np.sum(v2 * v2)))
    angle = np.arccos(angle) / 3.14 * 180

    return angle


def get_str_guester(up_fingers, list_lms):
    # 首先检查是否可能是OK手势
    thumb_tip = list_lms[4]
    index_tip = list_lms[8]
    distance = np.sqrt(np.sum((thumb_tip - index_tip) ** 2))
    
    # 如果大拇指和食指捏合，且中指、无名指和小指伸直，则为OK手势
    other_fingers = [12, 16, 20]
    if distance < 20 and all(finger in up_fingers for finger in other_fingers):
        return "OK"

    # 其他手势的判断
    if len(up_fingers) == 1 and up_fingers[0] == 8:
        str_guester = "1"

    elif len(up_fingers) == 1 and up_fingers[0] == 4:
        str_guester = "8"

    elif len(up_fingers) == 2 and up_fingers[0] == 8 and up_fingers[1] == 12:
        str_guester = "2"

    elif len(up_fingers) == 3 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16:
        str_guester = "3"

    elif len(up_fingers) == 4 and up_fingers[0] == 8 and up_fingers[1] == 12 and up_fingers[2] == 16 and up_fingers[3] == 20:
        str_guester = "4"

    elif len(up_fingers) == 5:
        str_guester = "5"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 20:
        str_guester = "6"

    elif len(up_fingers) == 2 and up_fingers[0] == 4 and up_fingers[1] == 8:
        # 如果距离小于阈值，认为是捏合手势7，否则是手势8
        if distance < 20:  # 可以根据实际情况调整阈值
            str_guester = "7"
        else:
            str_guester = "8"

    elif len(up_fingers) == 0:
        str_guester = "0"

    else:
        str_guester = "None"

    return str_guester


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    # 定义手 检测对象
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    while True:

        # 读取一帧图像
        success, img = cap.read()
        if not success:
            continue
        image_height, image_width, _ = np.shape(img)

        # 转换为RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 得到检测结果
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

            # 采集所有关键点的坐标
            list_lms = []
            for i in range(21):
                pos_x = hand.landmark[i].x * image_width
                pos_y = hand.landmark[i].y * image_height
                list_lms.append([int(pos_x), int(pos_y)])

            # 构造凸包点
            list_lms = np.array(list_lms, dtype=np.int32)
            hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17, 10]
            hull = cv2.convexHull(list_lms[hull_index, :])
            # 绘制凸包
            cv2.polylines(img, [hull], True, (0, 255, 0), 2)

            # 查找外部的点数
            n_fig = -1
            ll = [4, 8, 12, 16, 20]
            up_fingers = []

            for i in ll:
                pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < 0:
                    up_fingers.append(i)

            # print(up_fingers)
            # print(list_lms)
            # print(np.shape(list_lms))
            str_guester = get_str_guester(up_fingers, list_lms)

            cv2.putText(img, ' %s' % (str_guester), (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4,
                        cv2.LINE_AA)

            for i in ll:
                pos_x = hand.landmark[i].x * image_width
                pos_y = hand.landmark[i].y * image_height
                # 画点
                cv2.circle(img, (int(pos_x), int(pos_y)), 3, (0, 255, 255), -1)

        cv2.imshow("hands", img)

        key = cv2.waitKey(1) & 0xFF

        # 按键 "q" 退出
        if key == ord('q'):
            break
    cap.release()