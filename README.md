# Photo2Scan
Quick Software to convert images to scans

// Python reference

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



## consts
process_size = (250, 250)
low_white = np.array([120,120,120], dtype=np.uint8)
high_white = np.array([255,255,255], dtype=np.uint8)

offset = 25


## process

img = cv.imread("test2.jpg")
height, width, _ = img.shape

ratio = (height / process_size[0], width / process_size[1])
resized = cv.resize(img, process_size)
color_mask = cv.inRange(resized, low_white, high_white)


### some functions

def warp_perspective(img, src, dst):
    M = cv.getPerspectiveTransform(src, dst)
    warped_image = cv.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv.INTER_LINEAR)
    return warped_image

def counter_valid(counter, minc, max_c):
    if minc <= counter <= max_c:
        return True
    return False

def find_edge(arr, inc = True):
    
    threshold = max(arr) * 0.7
    
    if inc:
        counter = 0
    else:
        counter = len(arr) - 1

    while (counter_valid(counter, 0, len(arr) - 1)):
        if arr[counter] < threshold:
            if inc:
                counter += 1
            else:
                counter -= 1
        else:
            break
    if counter_valid(counter, 0, len(arr) - 1):
        return counter

    return -1

def get_slice(val, maxlen, offset):
    
    x = 0
    y = 0
    
    if val >= offset:
        x = val - offset
    else:
        x = 0
    
    if val + offset <= maxlen:
        y = val + offset
    else:
        y = maxlen
    
    return (x,y)


def generate_corner(color_mask, vert, height, hori, width):

	v = get_slice(vert, height, offset)
	h = get_slice(hori, width, offset)

	sliced = color_mask[v[0]:v[1], h[0]:h[1]]

	vert = np.sum(sliced, 1)
	hori = np.sum(sliced, 0)

	x = v[0] + find_edge(vert, True)
	y = h[0] + find_edge(hori, True)

	return (x,y)


# das kann noch arbeit vertragen
def generate_slices(color_mask, v_start, v_end, h_start, h_end):
    
    l = []
    offset = 25
    
    height, width = color_mask.shape

    
    # top left done
    tl_v = get_slice(v_start, height, offset)
    tl_h = get_slice(h_start, width, offset)
    
    tl_img = color_mask[tl_v[0]:tl_v[1],tl_h[0]:tl_h[1]]
    
    vert = np.sum(tl_img, 1)
    hori = np.sum(tl_img, 0)
    
    l.append(
        (tl_v[0] + find_edge(vert, True), tl_h[0] + find_edge(hori, True))
    )
    
    # top right done
    tl_v = get_slice(v_start, height, offset)
    tl_h = get_slice(h_end, width, offset)
    
    tl_img = color_mask[tl_v[0]:tl_v[1],tl_h[0]:tl_h[1]]
    
    vert = np.sum(tl_img, 1)
    hori = np.sum(tl_img, 0)
    
    l.append(
        (tl_v[0] + find_edge(vert, True), tl_h[0] + find_edge(hori, False))
    )
    
    # bot left done
    tl_v = get_slice(v_end, height, offset)
    tl_h = get_slice(h_start, width, offset)
    
    tl_img = color_mask[tl_v[0]:tl_v[1],tl_h[0]:tl_h[1]]
    
    vert = np.sum(tl_img, 1)
    hori = np.sum(tl_img, 0)
    
    l.append(
        (tl_v[0] + find_edge(vert, False), tl_h[0] + find_edge(hori, True))
    )
    
    # bot right done
    tl_v = get_slice(v_end, height, offset)
    tl_h = get_slice(h_end, width, offset)
    
    tl_img = color_mask[tl_v[0]:tl_v[1],tl_h[0]:tl_h[1]]
    
    vert = np.sum(tl_img, 1)
    hori = np.sum(tl_img, 0)
    
    l.append(
        (tl_v[0] + find_edge(vert, False), tl_h[0] + find_edge(hori, False))
    )
    
    return l

# estimate vert
vert = np.sum(color_mask, 1)
v_start = find_edge(vert, True)
v_end = find_edge(vert, False)

# estimate hori
hori = np.sum(color_mask, 0)
h_start = find_edge(hori, True)
h_end = find_edge(hori, False)


org_pts = generate_slices(color_mask, v_start, v_end, h_start, h_end)
    
edges = np.flip(np.float32(org_pts) * ratio,1)

dst = np.float32([
    [0,0],
    [width, 0],
    [0, height],
    [width, height]], dtype = "float32") # top right


final_img = warp_perspective(img, np.float32(edges), dst)

processed_img = cv.polylines(img,[np.int32([edges[0],
edges[1],
edges[3],
edges[2]])], True, (0,255,255), 20, cv.LINE_8)
plt.imshow(img)
