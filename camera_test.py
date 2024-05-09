import numpy as np
import cv2
import depthai as dai
from collections import namedtuple

Box = namedtuple('Box', ['x', 'y', 'w', 'h'])
Point2D = namedtuple('Point2D', ['x', 'y'])
Point3D = namedtuple('Point3D', ['x', 'y', 'z'])

# ==================== Settings ====================
# q - quit; w s a d - move box;
FPS = 15
LEFT_BOX = Box(120, 100, 150, 75)
UNDISTORT_RGB = True
# show Right cam and template matching:
SHOW_MORE_IMAGES = False
# for testing, use intrinsics for principle point or use resolution // 2:
# since the image are undistorted, I guess this should be False
USE_INTRINSICS_PRINCIPLE_POINT = False
# ==================================================

RGB, LEFT, RIGHT = 'rgb', 'left', 'right'

print(f'UNDISTORT_RGB: {UNDISTORT_RGB}')
print(f'USE_INTRINSICS_PRINCIPLE_POINT: {USE_INTRINSICS_PRINCIPLE_POINT}')
print()

def draw_box(img, p1, p2, label="", color=(255, 255, 255), line_thickness=3):
  p1, p2 = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
  tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
  cv2.rectangle(img, p1, p2, [0, 0, 0], thickness=tl + 1, lineType=4)
  cv2.rectangle(img, p1, p2, color, thickness=tl, lineType=4)
  cv2.circle(img, ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2), tl + 1, [0, 0, 0], thickness=-1)
  cv2.circle(img, ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2), tl, color, thickness=-1)
  tf = max(tl - 1, 1)
  for label in label.split('\n'):
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    p2 = p1[0] + t_size[0], p1[1] + t_size[1] + 10
    cv2.putText(img, label, (p1[0], p2[1] - 8), 0, tl / 3,[0, 0, 0], thickness=tf + 2, lineType=4)
    cv2.putText(img, label, (p1[0], p2[1] - 8), 0, tl / 3,[255, 255, 255], thickness=tf, lineType=4)
    p1 = p1[0], p2[1]


def getMesh(calibData, ispSize, camSocket):
  # getMesh as mentioned in my post
  M1 = np.array(calibData.getCameraIntrinsics(camSocket, ispSize[0], ispSize[1]))
  d1 = np.array(calibData.getDistortionCoefficients(camSocket))
  extrinsics = np.float64(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, camSocket))
  R1 = extrinsics[:3,:3]
  mapX, mapY = cv2.initUndistortRectifyMap(M1, d1, R1, M1, ispSize, cv2.CV_32FC1)
  meshCellSize = 16
  mesh0 = []
  for y in range(mapX.shape[0] + 1):
    if y % meshCellSize == 0:
      rowLeft = []
      for x in range(mapX.shape[1]):
        if x % meshCellSize == 0:
          if y == mapX.shape[0] and x == mapX.shape[1]:
            rowLeft.append(mapX[y - 1, x - 1])
            rowLeft.append(mapY[y - 1, x - 1])
          elif y == mapX.shape[0]:
            rowLeft.append(mapX[y - 1, x])
            rowLeft.append(mapY[y - 1, x])
          elif x == mapX.shape[1]:
            rowLeft.append(mapX[y, x - 1])
            rowLeft.append(mapY[y, x - 1])
          else:
            rowLeft.append(mapX[y, x])
            rowLeft.append(mapY[y, x])
      if (mapX.shape[1] % meshCellSize) % 2 != 0:
        rowLeft.append(0)
        rowLeft.append(0)
      mesh0.append(rowLeft)
  mesh0 = np.array(mesh0)
  meshWidth = mesh0.shape[1] // 2
  meshHeight = mesh0.shape[0]
  mesh0.resize(meshWidth * meshHeight, 2)
  mesh = list(map(tuple, mesh0))
  return mesh, meshWidth, meshHeight


pipeline = dai.Pipeline()
cams = {RGB: pipeline.create(dai.node.ColorCamera), LEFT: pipeline.create(dai.node.MonoCamera), RIGHT: pipeline.create(dai.node.MonoCamera)}
xouts = {name: pipeline.createXLinkOut() for name in cams.keys()}

# ==================== RGB pipeline ====================
cams[RGB].setBoardSocket(dai.CameraBoardSocket.CAM_A)  # rgb
cams[RGB].setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
if UNDISTORT_RGB:
  manip = pipeline.create(dai.node.ImageManip)
  manip.setMaxOutputFrameSize(cams[RGB].getIspWidth() * cams[RGB].getIspHeight() * 3 // 2)
  manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
  cams[RGB].isp.link(manip.inputImage)
  manip.out.link(xouts[RGB].input)
else:
  cams[RGB].video.link(xouts[RGB].input)
# ==================== Stereo pipeline ====================
cams[LEFT].setBoardSocket(dai.CameraBoardSocket.CAM_B)  # left
cams[LEFT].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

cams[RIGHT].setBoardSocket(dai.CameraBoardSocket.CAM_C)  # right
cams[RIGHT].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

stereo = pipeline.create(dai.node.StereoDepth)
cams[LEFT].out.link(stereo.left)
cams[RIGHT].out.link(stereo.right)
stereo.setRectifyEdgeFillColor(-1)
stereo.rectifiedLeft.link(xouts[LEFT].input)
stereo.rectifiedRight.link(xouts[RIGHT].input)
# ==================== Common pipeline settings ====================
for cam in cams.values():
  cam.setFps(FPS)

for name, xout in xouts.items():
  xout.setStreamName(name)
  xout.input.setBlocking(False)
  xout.input.setQueueSize(1)

# ==================== Boot camera ====================
while True:
  try:
    device = dai.Device()
  except RuntimeError as error:
    print(f'{error.args[0]}. Trying again...')
  else:
    break
# ==================== Mesh and Camera info ====================
calibration = device.readCalibration()
if UNDISTORT_RGB:
  mesh, meshWidth, meshHeight = getMesh(calibration, cams[RGB].getIspSize(), dai.CameraBoardSocket.CAM_A)
  manip.setWarpMesh(mesh, meshWidth, meshHeight)
left_right_baseline = calibration.getCameraExtrinsics(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_B)[0][3] / 100  # in mm
left_rgb_baseline   = calibration.getCameraExtrinsics(dai.CameraBoardSocket.CAM_A, dai.CameraBoardSocket.CAM_B)[0][3] / 100  # in mm
left_intrinsics = calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, cams[LEFT].getResolutionWidth(), cams[LEFT].getResolutionHeight())
rgb_intrinsics  = calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, cams[RGB].getResolutionWidth(),  cams[RGB].getResolutionHeight())
left_focal_len = left_intrinsics[0][0] # focal_x == focal_y
rgb_focal_len = rgb_intrinsics[0][0]   # focal_x == focal_y
if USE_INTRINSICS_PRINCIPLE_POINT:
  left_principle_point = Point2D(left_intrinsics[0][2], left_intrinsics[1][2])
  rgb_principle_point = Point2D(rgb_intrinsics[0][2], rgb_intrinsics[1][2])
else:
  left_principle_point = Point2D(cams[LEFT].getResolutionWidth() / 2, cams[LEFT].getResolutionHeight() / 2)
  rgb_principle_point = Point2D(cams[RGB].getResolutionWidth() / 2, cams[RGB].getResolutionHeight() / 2)

# Modify principle point as mentioned in my post:
# rgb_principle_point = Point2D(rgb_principle_point.x * 1.08, rgb_principle_point.y * 1.03)

print("camera info:")
print(f'left -> right baseline: {left_right_baseline}m;')
print(f'left -> rgb baseline:   {left_rgb_baseline}m;')
print(f'left_focal_x: {left_intrinsics[0][0]}; left_focal_y: {left_intrinsics[1][1]};')
print(f'rgb_focal_x:  {rgb_intrinsics[0][0]}; rgb_focal_y: {rgb_intrinsics[1][1]};')
print(f'left_principle_point: {left_principle_point};')
print(f'rgb_principle_point:  {rgb_principle_point};')
print()
# ==================== Main loop start ====================
device.startPipeline(pipeline)
queues = [device.getOutputQueue(name=name, maxSize=1, blocking=False) for name in xouts.keys()]
left_box = LEFT_BOX
while True:
  # ============ Get images ============
  imgs = {name: queue.get().getCvFrame() for queue, name in zip(queues, xouts.keys())}

  # ============ template match ============
  template = imgs[LEFT][left_box.y: left_box.y + left_box.h, left_box.x: left_box.x + left_box.w]
  crop_right, crop_left = left_box.x + left_box.w, max(0, left_box.x - 200)
  crop_top, crop_bottom = left_box.y, left_box.y + left_box.h
  match = cv2.matchTemplate(imgs[RIGHT][crop_top: crop_bottom, crop_left: crop_right], template, cv2.TM_CCOEFF_NORMED)
  res = cv2.minMaxLoc(match)[3]
  res = (res[0] + crop_left, res[1] + crop_top)
  if SHOW_MORE_IMAGES:
    cv2.imshow(name, imgs[RIGHT])
    cv2.imshow('template', template)
    cv2.imshow('match', imgs[RIGHT][res[1]: res[1] + left_box.h, res[0]: res[0] + left_box.w])

  # ============ Get 3D point ============
  disparity = left_box.x - res[0]
  if disparity == 0:
    continue
  left_box_center = Point2D(left_box.x + left_box.w // 2, left_box.y + left_box.h // 2)
  u = left_box_center.x - left_principle_point.x
  v = left_box_center.y - left_principle_point.y
  Z = left_focal_len * left_right_baseline / disparity
  X = u * Z / left_focal_len
  Y = v * Z / left_focal_len
  point_3d = Point3D(X, Y, Z)
  # ============ Draw left cam box ============
  bottom_corner = (left_box.x + left_box.w, left_box.y + left_box.h)
  label = f'X: {X:.2f}m\nY: {Y:.2f}m\nZ: {Z:.2f}m\nD: {disparity}\n'
  draw_box(imgs[LEFT], left_box, bottom_corner, label, line_thickness=1)
  cv2.imshow("left", imgs[LEFT])

  # ============ project 3D point to RGB ============
  point_3d_rgb = Point3D(point_3d.x - left_rgb_baseline, point_3d.y, point_3d.z)
  focal_scale = rgb_focal_len / left_focal_len
  rgb_w  = left_box.w * focal_scale
  rgb_h = left_box.h * focal_scale
  u = point_3d_rgb.x * rgb_focal_len / point_3d_rgb.z - rgb_w / 2
  v = point_3d_rgb.y * rgb_focal_len / point_3d_rgb.z - rgb_h / 2
  x = rgb_principle_point.x + u
  y = rgb_principle_point.y + v
  rgb_box = Box(*map(int, (x, y, rgb_w, rgb_h)))
  # ============ Draw rgb cam box ============
  bottom_corner = (rgb_box.x + rgb_box.w, rgb_box.y + rgb_box.h)
  draw_box(imgs[RGB], rgb_box, bottom_corner, line_thickness=round(focal_scale))
  cv2.imshow("rgb", imgs[RGB])

  # ============ Controls ============
  if (key:=cv2.waitKey(1)) == ord('q'):
    break
  elif key == ord('w'):
    left_box = Box(left_box.x, left_box.y - 5, left_box.w, left_box.h)
  elif key == ord('s'):
    left_box = Box(left_box.x, left_box.y + 5, left_box.w, left_box.h)
  elif key == ord('a'):
    left_box = Box(left_box.x - 5, left_box.y, left_box.w, left_box.h)
  elif key == ord('d'):
    left_box = Box(left_box.x + 5, left_box.y, left_box.w, left_box.h)
