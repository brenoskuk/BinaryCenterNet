#############################################################
#
#   POSITION EST FUNCS
#
#############################################################

def calculate_error(estimated_pos, gt_pos):
    return np.linalg.norm(np.array(estimated_pos) - np.array(gt_pos))/np.linalg.norm(np.array(gt_pos))*100

def plot_result_position(idx, imgs, anno, positions):
    img = imgs[idx]
    car_img_coords_a = get_bb_center(anno[idx]['bboxes'][0])
    car_img_coords_b = get_bb_center(anno[idx]['bboxes'][1])

    car_xy_coords_a = positions[idx][0]
    car_xy_coords_b = positions[idx][1]

    pos_car_a = get_vehicle_pos(car_img_coords_a)
    pos_car_b = get_vehicle_pos(car_img_coords_b)

    plot_2_cars_center(img, car_img_coords_a, car_img_coords_b)

    estimated_a_coords = np.array([camera_coords[0] + pos_car_a[0], camera_coords[1] -pos_car_a[1]])
    estimated_b_coords = np.array([camera_coords[0] + pos_car_b[0], camera_coords[1] -pos_car_b[1]])

    pos_a_error = calculate_error(estimated_a_coords, car_xy_coords_a)
    pos_b_error = calculate_error(estimated_b_coords, car_xy_coords_b)


    print("Red dot: ")
    print("XY car coordinates relative to camera: ", (pos_car_a[0],pos_car_a[1]))
    print("Estimated world car coordinates: ", estimated_a_coords)
    print("Real coordinates: ", car_xy_coords_a)
    print("Error : {} %".format(pos_a_error))

    print("\n")

    print("Yellow dot: ")
    print("XY car coordinates relative to camera: ", (pos_car_b[0],pos_car_b[1]))
    print("Estimated world car coordinates: ", estimated_b_coords)
    print("Real coordinates: ", car_xy_coords_b)
    print("Error : {} %".format(pos_b_error))

    print("\n")

    print("Relative car distance: ", np.linalg.norm(estimated_a_coords - estimated_b_coords), ' [m]')

def get_bb_center(bb):
    """
     calculate object center based upon it's bounding box coordinates
    :bb: image coordinates of bounding box are assumed to be first 4 entries of bb
    :return: center of bounding box 
    """     
    ul = (bb[0], bb[1])
    lr = (bb[2], bb[3])
    center_coord = ( ul[0] + (lr[0] - ul[0])/2, ul[1] + (lr[1] - ul[1])/2)
    return center_coord

def plot_object_center(img, prediction, figsize = (10,10)):
    plt.figure(figsize=figsize)
    
    
    # plot bbox
    plt.plot(prediction[0],prediction[1], 'x', c = 'y', markersize = '10')
    plt.plot(prediction[2],prediction[3], 'x', c = 'y', markersize = '10')
    
    
    # plot bbox center
    center_coords = get_bb_center(prediction)
    plt.plot(center_coords[0],center_coords[1], 'o', c = 'r', markersize = '10')

    plt.imshow(img)
    plt.show()

# camera resolution in pixels
HEIGHT_RES = 1080 # Image width in pixels
WIDTH_RES = 1920 # Image height in pixels

# center of image (in image coordinates)
CAM_CENTER =  (WIDTH_RES/2, HEIGHT_RES/2)

# field of view in degrees
FOV_H = 62.8 # FOV in degrees
FOV_W = 36.8 # FOV in degrees

# camera tilt in degrees
W = 7.4

# camera altitude
CAMERA_ALTITUDE = 5

# calculate the focal lenght
FL_H = (HEIGHT_RES/2)/(np.tan(np.deg2rad(FOV_H/2)))
FL_W = (WIDTH_RES/2)/(np.tan(np.deg2rad(FOV_W/2)))

# coordinates of the camera in real world
CAMERA_COORDS = (0, 0)


def get_object_pos(obj_img_coords, camera_altitude = CAMERA_ALTITUDE, 
                   camera_tilt = W, cam_center = CAM_CENTER, fl_w = FL_H, fl_h= FL_W):
    """
     calculate object coordinates in the plane projected onto image
    :obj_img_coords: image coordinates of object 
    :camera_altitude: camera altitude in radial earth direction
    :camera_tilt: camera tilt wrp to orthogonal plane to radial earth direction
    :cam_center: camera center in pixel coordinates
    :fl_w: width focal lenght
    :fl_w: width focal lenght
    :fl_h: heigth focal lenght
    :return: relative coordinates to camera plane
    """ 
    
    # calculate heigth object angle in degrees
    theta_h = math.degrees(np.arctan((obj_img_coords[1] - cam_center[1])/fl_h))
    
    # account for tilt
    theta_h_tilt = theta_h + camera_tilt

    # calculate distance from car to camera
    I = camera_altitude/np.tan(math.radians(theta_h_tilt))

    # calculate heigth object angle
    tan_theta_w = (obj_img_coords[0]-cam_center[0])/(2*fl_w) # correct this later...

    # calculate distance of car to central axis 
    J = tan_theta_w*I

    return (I, J)
    