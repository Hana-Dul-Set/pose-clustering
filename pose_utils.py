import numpy as np

parent_pairs = [(0, 0), (1, 0), (2, 0), (17, 0), (18, 0), (3, 1), (4, 2), (5, 18), (6, 18), (19, 18), (11, 19), (12, 19), (7, 5), (8, 6), (9, 7), (10, 8), (13, 11), (14, 12), (15, 13), (16, 14), (24, 15), (25, 16), (20, 24), (22, 24), (21, 25), (23, 25)]


    


'''
Distance of person
- Face size
Position of pose in photo(Normalized by photo size)
- Keypoint 0(nose)'s position
The pose itself
- Each keypoint's angle relative to parent keypoint
- Each keypoint's position relative to parent keypoint (Normalized by pose square box)
'''
def keypoints2representation(keypoints, img_size):
    #face_size
    face_points = keypoints[26:94] + [keypoints[17]]
    left    = min(face_points, key=lambda x:x[0])[0]
    right   = max(face_points, key=lambda x:x[0])[0]
    top     = min(face_points, key=lambda x:x[1])[1]
    bottom  = max(face_points, key=lambda x:x[1])[1]
    face_size = (right-left) * (bottom - top)

    #nose_pos
    nose_pos = keypoints[0][:2]

    #relative positions
    pose = np.array([[key[0], key[1]] for key in keypoints[:26]])
    #pose = (pose - pose.min(axis = 0)) / (pose.max(axis = 0) - pose.min(axis = 0)) #pose bounding box
    pose = (pose - pose.min()) / (pose.max() - pose.min()) #pose square box
    pose -= pose[0]

    for pair in parent_pairs.__reversed__():
        pose[pair[0]] -= pose[pair[1]]
    return {'face_size':face_size, 'nose_pos': nose_pos, 'pose':pose}
