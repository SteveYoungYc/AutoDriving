from moviepy.editor import VideoFileClip
import numpy as np
import cv2

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 200
max_line_gap = 3.2

# roi_vtx = np.array([[(0, 720), (0, 380), (1280, 380), (1280, 720), (1050, 720), (780, 440), (500, 440), (230, 720)]])
roi_vtx = np.array([[(0, 720), (0, 380), (1280, 380), (1280, 720), (1050, 720), (780, 440), (500, 440), (230, 720),
                     (230, 380), (1050, 380)]])
ymin = 380
ymax = 720


def roi_mask(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) >= 3:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask_color = (255,) * channel_count
    else:
        mask_color = 255

    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], 2)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    try:
        draw_lanes(line_img, lines)
    except:
        pass
    return line_img


def draw_lanes(img, lines, color=(255, 0, 0), thickness=8):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if len(left_lines) == 0 or len(right_lines) == 0:
        return img

    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)

    left_points = [(x1, y1) for line in left_lines for x1, y1, _, _ in line]
    left_points = left_points + [(x2, y2) for line in left_lines for _, _, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, _, _ in line]
    right_points = right_points + [(x2, y2) for line in right_lines for _, _, x2, y2 in line]

    left_vtx = calc_lane_vertices(left_points)
    right_vtx = calc_lane_vertices(right_points)

    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)


def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def calc_lane_vertices(point_list):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return ((xmin, ymin), (xmax, ymax))


def process_an_image(img):
    blur_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_edges = roi_mask(edges, roi_vtx)
    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)

    return res_img


# 生成一个输出视频

output = 'D:\\SJTU\\Auto Driving\\lane-out.mp4'  # ouput video
clip = VideoFileClip("D:\\SJTU\\Auto Driving\\20200618 124512_outVideo.avi")  # input video
out_clip = clip.fl_image(process_an_image)  # 对视频的每一帧进行处理
out_clip.write_videofile(output, audio=True)  # 将处理后的视频写入新的视频文件


# 实时显示
cap = cv2.VideoCapture(
    "D:\\SJTU\\Auto Driving\\20200618 124512_outVideo.avi")  # 指定路径读取视频。如果cv2.VideoCapture(0)，没有指定路径，则从电脑自带摄像头取视频。
ret = True
# ret,frame = cap.read()
while ret:
    ret, frame = cap.read()  # 按帧读取视频，它的返回值有两个：ret, frame。其中ret是布尔值，如果读取帧是正确的则返回 True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
    if ret:
        cv2.imshow('Imge', frame)  # 播放视频，第一个参数是视频播放窗口的名称，第二个参数是视频的当前帧。
        res_img = process_an_image(frame)
        cv2.imshow('AQ', res_img)
    k = cv2.waitKey(20)  # 每一帧的播放时间，毫秒级,该参数可以根据显示速率调整
    if k & 0xff == ord('q'):  # 如果中途想退出，q键退出，或播放完后，按任意键退出
        cap.release()
        cv2.destroyAllWindows()  # 释放对象和销毁窗口
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
