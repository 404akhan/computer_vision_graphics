import numpy as np
import cv2
import os
from time import gmtime, strftime
import scipy.sparse.linalg as sla


# global vars, init
resize_fac = 2
points = []
font = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread('images/box6.bmp')
img_clean_copy = np.copy(img)

print 'img resized by %d\n' % resize_fac
img = cv2.resize(img, (0, 0), fx=resize_fac, fy=resize_fac)

state = 'define_pts'
van_pts = [] # x, y, z
alphas = []

pos_3d = {}
regions = []


# functions
def load_pts():
	# only for demo
	global img, points

	if state != 'define_pts':
		return

	img = np.copy(img_clean_copy)
	points = [
		np.array([162,227]), 
		np.array([174,366]), 
		np.array([391,542]), 
		np.array([393,399]), 
		
		np.array([606,431]), 
		np.array([618,290]), 
		np.array([379,139]), 
		np.array([382,277])
	]

	for index, pt in enumerate(points):
		cv2.circle(img, tuple(pt), 5, (0,0,255), -1)	
		cv2.putText(img, '%d'%index, tuple(pt), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

		print 'point %d, coords: (%d, %d)' % (index, pt[0], pt[1])


def draw_circle(event,x,y,flags,param):
	global points, img, state

	if state != 'define_pts':
		return

	if event == cv2.EVENT_LBUTTONDBLCLK:
		
		cv2.circle(img, (x, y), 5, (0,0,255), -1)	
		points.append(np.array([x, y]))

		index = len(points) - 1
		cv2.putText(img, '%d'%index, (x, y), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
		
		print 'point %d, coords: (%d, %d)' % (index, x, y)


def inters_lines(lines):
	M = np.zeros((3, 3))

	for line in lines:
		l_p1 = np.append(line[0], 1)
		l_p2 = np.append(line[1], 1)

		l_h = np.cross(l_p1, l_p2)

		l_vec = np.reshape(l_h, (1, 3))
		M = M + np.transpose(l_vec).dot(l_vec)

	eigenValues, eigenVectors = sla.eigs(M, k=1, which='SM')
	eigenVectors = np.transpose(eigenVectors.real)
	
	# Convert coordinates into homogeneous form
	V = (eigenVectors[-1]/eigenVectors[-1,-1])
	V = V.astype(int)

	return V[:2]


# def inters_lines(lines):
# 	# TODO if lines > 2
# 	# li | [p1, p2]; where pi 2d array
# 	# return ip intersection point of l1, l2
# 	l1, l2 = lines

# 	l1_p1 = np.append(l1[0], 1)
# 	l1_p2 = np.append(l1[1], 1)

# 	l2_p1 = np.append(l2[0], 1)
# 	l2_p2 = np.append(l2[1], 1)

# 	l1_h = np.cross(l1_p1, l1_p2)
# 	l2_h = np.cross(l2_p1, l2_p2)

# 	ip = np.cross(l1_h, l2_h)
# 	ip = ip/ip[2]
# 	ip = ip[:2]

# 	print ip
# 	return ip


def draw_line(p1, p2):
	cv2.line(img, tuple(p1), tuple(p2), (255,0,0), 2)


def get_alpha1(b, t, vz, ln, z):
	top = - np.linalg.norm( np.cross(b, t))
	bot = np.dot(ln, b) * np.linalg.norm( np.cross(vz, t))
	alphaz = top / (bot * z)

	return alphaz


def get_alphaz(orig, pz, dz):
	global van_pts
	van_pts_loc = []

	for i in range(3):
		van_pts_loc.append(np.append(van_pts[i], 1))

	vx, vy, vz = van_pts_loc
	crz = np.cross(vx, vy)
	ln = crz / np.linalg.norm(crz)

	b = np.append(orig, 1)
	t = np.append(pz, 1)

	alphaz = get_alpha1(b, t, vz, ln, dz)

	return alphaz


def get_alphay(orig, py, dy):
	global van_pts
	van_pts_loc = []
	
	for i in range(3):
		van_pts_loc.append(np.append(van_pts[i], 1))

	vx, vy, vz = van_pts_loc
	cry = np.cross(vx, vz)
	ln = cry / np.linalg.norm(cry)

	b = np.append(orig, 1)
	t = np.append(py, 1)

	alphay = get_alpha1(b, t, vy, ln, dy)

	return alphay


def get_alphax(orig, px, dx):
	global van_pts
	van_pts_loc = []
	
	for i in range(3):
		van_pts_loc.append(np.append(van_pts[i], 1))

	vx, vy, vz = van_pts_loc
	crx = np.cross(vy, vz)
	ln = crx / np.linalg.norm(crx)

	b = np.append(orig, 1)
	t = np.append(px, 1)

	alphax = get_alpha1(b, t, vx, ln, dx)

	return alphax


def update_3d_pos(direction, p1, p2):
	global van_pts, pos_3d
	van_pts_loc = []

	for i in range(3):
		van_pts_loc.append(np.append(van_pts[i], 1))
	vx, vy, vz = van_pts_loc

	if direction == 'z':
		b = np.append(points[p1], 1)
		t = np.append(points[p2], 1)
		
		crz = np.cross(vx, vy)
		ln = crz / np.linalg.norm(crz)

		alphaz = alphas[2]

		top = np.linalg.norm( np.cross(b, t))
		bot = np.dot(ln, b) * np.linalg.norm( np.cross(vz, t))
		deltaZ = top / (bot * alphaz)

		z1 = pos_3d[p1][2]
		z2 = z1 - deltaZ

		x2 = pos_3d[p1][0]
		y2 = pos_3d[p1][1]
		pos_3d[p2] = [x2, y2, z2]


	if direction == 'y':
		b = np.append(points[p1], 1)
		t = np.append(points[p2], 1)

		cry = np.cross(vx, vz)
		ln = cry / np.linalg.norm(cry)
		
		alphay = alphas[1]

		top = np.linalg.norm( np.cross(b, t))
		bot = np.dot(ln, b) * np.linalg.norm( np.cross(vy, t))
		deltaY = top / (bot * alphay)

		y1 = pos_3d[p1][1]
		y2 = y1 - deltaY

		x2 = pos_3d[p1][0]
		z2 = pos_3d[p1][2]
		pos_3d[p2] = [x2, y2, z2]


	if direction == 'x':
		b = np.append(points[p1], 1)
		t = np.append(points[p2], 1)

		crx = np.cross(vy, vz)
		ln = crx / np.linalg.norm(crx)

		alphax = alphas[0]

		top = np.linalg.norm( np.cross(b, t))
		bot = np.dot(ln, b) * np.linalg.norm( np.cross(vx, t))
		deltaX = top / (bot * alphax)

		x1 = pos_3d[p1][0]
		x2 = x1 - deltaX

		y2 = pos_3d[p1][1]
		z2 = pos_3d[p1][2]
		pos_3d[p2] = [x2, y2, z2]


def get_warpImage(screenCnt, image):
	orig = image.copy()	
	rect = screenCnt.reshape(4, 2).astype('float32')

	# now that we have our rectangle of points, let's compute
	# the width of our new image
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

	# ...and now for the height of our new image
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

	# take the maximum of the width and height values to reach
	# our final dimensions
	maxWidth = max(int(widthA), int(widthB))
	maxHeight = max(int(heightA), int(heightB))

	# construct our destination points which will be used to
	# map the screen to a top-down, "birds eye" view
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# calculate the perspective transform matrix and warp
	# the perspective to grab the screen
	M = cv2.getPerspectiveTransform(rect, dst)
	warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

	return warp


def create_vrml():
	global regions

	path = strftime("%Y-%m-%d---%H:%M:%S", gmtime())
	os.makedirs(path)
	wr_gl = """#VRML V2.0 utf8

	"""

	for ind, [q, warp] in enumerate(regions):
		cv2.imwrite('%s/warp%d.jpg'%(path, ind), warp)
		
		wr = """
			Shape {
			 appearance Appearance {
				 material Material { }
				 texture ImageTexture {
					 url "warp%d.jpg"
				 }
			 }
			 geometry IndexedFaceSet {
				 coord Coordinate {
					 point [
						 %d %d %d,
						 %d %d %d,
						 %d %d %d,
						 %d %d %d,
					 ]
				 }
				 coordIndex [
					 0, 1, 2, 3, -1,
				 ]
				 texCoord TextureCoordinate {
					 point [
						 0 0,
						 1 0,
						 1 1,
						 0 1,
					 ]
				 }
				 texCoordIndex [
					 0, 1, 2, 3, -1,
				 ]
				 solid FALSE
			 }
			}

		"""

		wr = wr % (ind, q[3][0], q[3][1], q[3][2], q[2][0], q[2][1], q[2][2], 
						q[1][0], q[1][1], q[1][2], q[0][0], q[0][1], q[0][2])
		wr_gl += wr

	text_file = open("%s/out_vrml.wrl" % path, "w")
	text_file.write(wr_gl)
	text_file.close()


# start
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)


print 'define points:'

while(1):
	cv2.imshow('image', img)
	k = cv2.waitKey(20) & 0xFF	
	
	# close
	if k == 27:
		break

	
	if k == ord('l'):
		load_pts()


	if k == ord('v'):
		state = 'calc_van_pts'

		cur_van_dir = 'x'
		if len(van_pts) == 1: cur_van_dir = 'y'
		if len(van_pts) == 2: cur_van_dir = 'z'

		if len(van_pts) == 3: 
			print 'all van pts are defined'
			continue

		print '\nCalculate vanishing points along %s' % cur_van_dir
		lines_input = raw_input('insert lines as list (even number): p1 p2 p3 p4 ...\n')

		pts = lines_input.split()
		pt_indexes = [int(pt) for pt in pts]		

		lines = []
		for i in range(len(pt_indexes)/2):
			ind1, ind2 = pt_indexes[2*i], pt_indexes[2*i+1]
			lines.append([points[ind1], points[ind2]])

			draw_line(points[ind1], points[ind2])


		van_p = inters_lines(lines)
		print 'vanishing point along %s: (%d, %d)' % (cur_van_dir, van_p[0], van_p[1])

		for line in lines:
			# for debug
			draw_line(line[0], van_p)

		
		van_pts.append(van_p)
		if len(van_pts) == 3: 
			print van_pts
			state = 'calc_alphas'


	if k == ord('c'):
		if state == 'calc_alphas':
			# dz is distance from origin to pz

			print '\nTo know origin and alphas'
			loc_input = raw_input('input: orig px dx py dy pz dz\n')

			arr_input = loc_input.split()
			int_input = [int(inp) for inp in arr_input]	

			orig = points[int_input[0]]
			px = points[int_input[1]]
			py = points[int_input[3]]
			pz = points[int_input[5]]

			dx = int_input[2]
			dy = int_input[4]
			dz = int_input[6]

			alphax = get_alphax(orig, px, dx)
			alphay = get_alphay(orig, py, dy)
			alphaz = get_alphaz(orig, pz, dz)

			alphas = [alphax, alphay, alphaz]

			print alphas
			state = '3d_position'

			pos_3d[int_input[0]] = [0, 0, 0]
			pos_3d[int_input[1]] = [dx, 0, 0]
			pos_3d[int_input[3]] = [0, dy, 0]
			pos_3d[int_input[5]] = [0, 0, dz]


	if k == ord('p'):
		if state == '3d_position':
			print '\ncurrently know positions: id-> (x, y, z)'
			print pos_3d

			print 'find 3d position, insert direction, point with known pos, another point along direction'
			loc_input = raw_input('input: dir p1 p2\n')

			arr_input = loc_input.split()

			direction = arr_input[0]
			p1_id = int(arr_input[1])
			p2_id = int(arr_input[2])

			update_3d_pos(direction, p1_id, p2_id)
			print 'computed position for', p2_id, pos_3d[p2_id]

			if len(pos_3d) == len(points):
				state = 'comp_text_map'


	if k == ord('t'):
		if state == 'comp_text_map':
			print '\n enter quadrilaterals for texture mapping'
			loc_input = raw_input('input, order [0,0], [1,0], [1,1], [0,1]: p1 p2 p3 p4\n')

			arr_input = loc_input.split()
			int_input = [int(inp) for inp in arr_input]	

			img_loc = np.copy(img_clean_copy)
			screenCnt = np.array([points[i] for i in int_input])

			warp = get_warpImage(screenCnt, img_loc)

			quad_pts_3d = [pos_3d[i] for i in int_input]
			regions.append([quad_pts_3d, warp])
			
			cv2.imshow("warp", warp)
			cv2.waitKey(0)
			cv2.destroyWindow('warp')


	if k == ord('f'):
		if state == 'comp_text_map':
			create_vrml()

			print '\nFind VRML in created dir'
			print 'THANKS that\'s all!'

