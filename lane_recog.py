import cv2
import numpy as np
from moviepy.editor import VideoFileClip

# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20 

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshol

roi_vtx = np.array([[(0, 720), (0, 380), (1280, 380), (1280, 720), (1050, 720), (780, 440), (500, 440), (230, 720)]])
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
    return cv2.bitwise_and(img, mask)

def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, (255, 0, 0), thickness = 2)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength = min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    try:
        draw_lanes(line_img, lines)
    except:
        pass
    return line_img

def draw_lanes(img, lines, color = (255, 0, 0), thickness = 8):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (y2 - y1) / (x2 - x1) < 0:
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

def get_roi_edges(img):
    blur_gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (blur_ksize, blur_ksize), 0, 0)
    edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
    roi_edges = roi_mask(edges, roi_vtx)
    return roi_edges


# 生成一个输出视频
clip = VideoFileClip(r'video\20200618 124512_outVideo.avi')
out_clip = clip.fl_image(process_an_image)
out_clip.write_videofile(r'video\lane-out.mp4', audio = True)

