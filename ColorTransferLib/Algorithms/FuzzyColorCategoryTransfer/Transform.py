import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from .ColorSpace import ColorSpace


class Transform:

    color_terms = np.array(["Red", "Yellow", "Green", "Blue", "Black", "White", "Grey", "Orange", "Brown", "Pink", "Purple"])
    color_terms_id = {"Red":0, "Yellow":1, "Green":2, "Blue":3, "Black":4, "White":5, "Grey":6, "Orange":7, "Brown":8, "Pink":9, "Purple":10}

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rotation(class_pairs, color_cats_src):
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]
            if src_vol == 0.0 or ref_vol == 0.0:
                continue
            # get rotation angle
            src_xy = src_cen[:2]
            ref_xy = ref_cen[:2]

            # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
            dot = src_xy[0] * ref_xy[0] + src_xy[1] * ref_xy[1]
            det = src_xy[0] * ref_xy[1] - src_xy[1] * ref_xy[0]
            radians = math.atan2(det, dot)

            # Rotation in xy plane. z-Axis will be ignored
            x = color_cats_src[src_col][:,0]
            y = color_cats_src[src_col][:,1]
            xx = x * np.cos(radians) - y * np.sin(radians)
            yy = x * np.sin(radians) + y * np.cos(radians)

            color_cats_src[src_col][:,0] = xx
            color_cats_src[src_col][:,1] = yy
        return color_cats_src
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def scaling(class_pairs, color_cats_src, color_cats_ref):
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, ref_evec, ref_eval = elem[1]

            if src_vol == 0.0 or ref_vol == 0.0:
                continue

            # move temporarily to the origin for proper scaling
            rep = np.tile(ref_cen, (color_cats_src[src_col].shape[0],1))
            color_cats_src[src_col] -= rep
            rep = np.tile(ref_cen, (color_cats_ref[ref_col].shape[0],1))
            color_cats_ref[ref_col] -= rep

            # rotate source by eigenvectors of reference
            """
            rotation_ref = np.transpose(ref_evec)
            color_cats_src[src_col] = rotation_ref.dot(color_cats_src[src_col].T).T
            """

            # scale source to unit
            for r in range(50):
                ran_rotation = R.random().as_matrix()
                color_cats_src[src_col] = color_cats_src[src_col].dot(ran_rotation)
                color_cats_ref[ref_col] = color_cats_ref[ref_col].dot(ran_rotation)

                x_min, y_min, z_min = np.amin(color_cats_src[src_col], axis=0)
                x_max, y_max, z_max = np.amax(color_cats_src[src_col], axis=0)

                x_stretch = x_max - x_min
                y_stretch = y_max - y_min
                z_stretch = z_max - z_min
                stretch = np.array([x_stretch, y_stretch, z_stretch])
                scale_down = np.tile(stretch, (color_cats_src[src_col].shape[0],1))
                color_cats_src[src_col] = color_cats_src[src_col] / scale_down

                # scaling via eigenvectors and -values
                x_min, y_min, z_min = np.amin(color_cats_ref[ref_col], axis=0)
                x_max, y_max, z_max = np.amax(color_cats_ref[ref_col], axis=0)

                x_stretch = x_max - x_min
                y_stretch = y_max - y_min
                z_stretch = z_max - z_min
                stretch = np.array([x_stretch, y_stretch, z_stretch])
                scale_up = np.tile(stretch, (color_cats_src[src_col].shape[0],1))
                color_cats_src[src_col] = color_cats_src[src_col] * scale_up

                ran_rotation_inv = np.transpose(ran_rotation)
                color_cats_src[src_col] = color_cats_src[src_col].dot(ran_rotation_inv)
                color_cats_ref[ref_col] = color_cats_ref[ref_col].dot(ran_rotation_inv)
                """
                scale = np.tile(ref_eval, (color_cats_src[src_col].shape[0],1))
                color_cats_src[src_col] = color_cats_src[src_col] * scale
                """

            # move back to original position
            rep = np.tile(ref_cen, (color_cats_src[src_col].shape[0],1))
            color_cats_src[src_col] += rep
            rep = np.tile(ref_cen, (color_cats_ref[ref_col].shape[0],1))
            color_cats_ref[ref_col] += rep

        return color_cats_src
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_rotation_matrix(class_pairs):
        rots = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]
            if src_vol == 0.0 or ref_vol == 0.0:
                rots[src_col] = np.eye(4)
                continue

            # get rotation angle
            src_xy = src_cen[:2]
            ref_xy = ref_cen[:2]

            # https://stackoverflow.com/questions/14066933/direct-way-of-computing-clockwise-angle-between-2-vectors
            dot = src_xy[0] * ref_xy[0] + src_xy[1] * ref_xy[1]
            det = src_xy[0] * ref_xy[1] - src_xy[1] * ref_xy[0]
            radians = math.atan2(det, dot)

            rotation_mat = np.array([[np.cos(radians), -np.sin(radians), 0.0, 0.0],
                                     [np.sin(radians), np.cos(radians) , 0.0, 0.0],
                                     [0.0            , 0.0                , 1.0, 0.0],
                                     [0.0             , 0.0                , 0.0, 1.0]])

            rots[src_col] = rotation_mat
        return rots
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_translation_matrix(class_pairs, CV_src_new):
        trans = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            src_cen_new, src_vol_new = CV_src_new[src_col]

            if src_vol_new == 0.0 or ref_vol == 0.0:
                trans[src_col] = np.eye(4)
                continue

            translation = ref_cen - src_cen_new

            translation_mat = np.array([[1.0, 0.0, 0.0, translation[0]],
                                        [0.0, 1.0, 0.0, translation[1]],
                                        [0.0, 0.0, 1.0, translation[2]],
                                        [0.0, 0.0, 0.0, 1.0]])

            # translation_mat = np.array([[1.0, 0.0, 0.0, translation[0]],
            #                             [0.0, 1.0, 0.0, translation[1]],
            #                             [0.0, 0.0, 1.0, 0.0],
            #                             [0.0, 0.0, 0.0, 1.0]])

            trans[src_col] = translation_mat
        return trans

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def translation(class_pairs, color_cats_src, CV_src_new):
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            src_cen_new, src_vol_new = CV_src_new[src_col]

            if src_vol_new == 0.0 or ref_vol == 0.0:
                continue

            translation = ref_cen - src_cen_new

            rep = np.tile(translation, (color_cats_src[src_col].shape[0],1))

            color_cats_src[src_col] += rep
        return color_cats_src
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_scaling_matrix(class_pairs, color_cats_src, color_cats_ref):
        scal = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_cats_src_temp = copy.deepcopy(color_cats_src)
        color_cats_ref_temp = copy.deepcopy(color_cats_ref)

        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            if src_vol == 0.0 or ref_vol == 0.0:
                scal[src_col] = np.eye(4)
                continue

            
            # x_min, y_min, z_min = np.amin(color_cats_src_temp[src_col], axis=0)
            # x_max, y_max, z_max = np.amax(color_cats_src_temp[src_col], axis=0)

            # center_src = np.array([(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2])

            # trans_center = np.array([[1.0, 0.0, 0.0, -center_src[0]],
            #                          [0.0, 1.0, 0.0, -center_src[1]],
            #                          [0.0, 0.0, 1.0, -center_src[2]],
            #                          [0.0, 0.0, 0.0, 1.0        ]])
            
            # x_min, y_min, z_min = np.amin(color_cats_ref_temp[ref_col], axis=0)
            # x_max, y_max, z_max = np.amax(color_cats_ref_temp[ref_col], axis=0)

            # center_ref = np.array([(x_max - x_min)/2, (y_max - y_min)/2, (z_max - z_min)/2])

            
            # rep = np.tile(center_src, (color_cats_src_temp[src_col].shape[0],1))
            # color_cats_src_temp[src_col] -= rep
            # rep = np.tile(center_ref, (color_cats_ref_temp[ref_col].shape[0],1))
            # color_cats_ref_temp[ref_col] -= rep

            trans_center = np.array([[1.0, 0.0, 0.0, -ref_cen[0]],
                                     [0.0, 1.0, 0.0, -ref_cen[1]],
                                     [0.0, 0.0, 1.0, -ref_cen[2]],
                                     [0.0, 0.0, 0.0, 1.0        ]])
            


            rep = np.tile(ref_cen, (color_cats_src_temp[src_col].shape[0],1))
            color_cats_src_temp[src_col] -= rep
            rep = np.tile(ref_cen, (color_cats_ref_temp[ref_col].shape[0],1))
            color_cats_ref_temp[ref_col] -= rep

            total_scaling = np.eye(4)

            for r in range(50):
                ran_rotation = R.random().as_matrix()
                ran_rot_mat = np.eye(4)
                ran_rot_mat[:3,:3] = ran_rotation


                color_cats_src_temp[src_col] = Transform.transform_single(color_cats_src_temp[src_col], ran_rot_mat)

                #color_cats_ref_temp[ref_col] = color_cats_ref_temp[ref_col].dot(ran_rotation)
                color_cats_ref_temp[ref_col] = Transform.transform_single(color_cats_ref_temp[ref_col], ran_rot_mat)

                x_min, y_min, z_min = np.amin(color_cats_src_temp[src_col], axis=0)
                x_max, y_max, z_max = np.amax(color_cats_src_temp[src_col], axis=0)

                x_stretch = x_max - x_min
                y_stretch = y_max - y_min
                z_stretch = z_max - z_min
                scale_down = np.array([1.0/x_stretch, 1.0/y_stretch, 1.0/z_stretch])
                
                scale_down_m = np.tile(scale_down, (color_cats_src_temp[src_col].shape[0],1))
                color_cats_src_temp[src_col] = color_cats_src_temp[src_col] * scale_down_m

                #scale_down_mat = np.eye(4)
                scale_down_mat = np.array([[scale_down[0], 0.0            , 0.0            , 0.0],
                                           [0.0            , scale_down[1], 0.0            , 0.0],
                                           [0.0            , 0.0            , scale_down[2], 0.0],
                                           [0.0            , 0.0            , 0.0            , 1.0]])


                x_min, y_min, z_min = np.amin(color_cats_ref_temp[ref_col], axis=0)
                x_max, y_max, z_max = np.amax(color_cats_ref_temp[ref_col], axis=0)

                x_stretch = x_max - x_min
                y_stretch = y_max - y_min
                z_stretch = z_max - z_min
                scale_up = np.array([x_stretch, y_stretch, z_stretch])

                scale_up_m = np.tile(scale_up, (color_cats_src_temp[src_col].shape[0],1))
                color_cats_src_temp[src_col] = color_cats_src_temp[src_col] * scale_up_m

                scale_up_mat = np.array([[scale_up[0], 0.0          , 0.0          , 0.0],
                                         [0.0          , scale_up[1], 0.0          , 0.0],
                                         [0.0          , 0.0          , scale_up[2], 0.0],
                                         [0.0          , 0.0          , 0.0          , 1.0]])
                

                ran_rotation_inv = np.transpose(ran_rotation)
                ran_rotinv_mat = np.eye(4)
                ran_rotinv_mat[:3,:3] = ran_rotation_inv

                # color_cats_src_temp[src_col] = color_cats_src_temp[src_col].dot(ran_rotation_inv)
                # color_cats_ref_temp[ref_col] = color_cats_ref_temp[ref_col].dot(ran_rotation_inv)
                color_cats_src_temp[src_col] = Transform.transform_single(color_cats_src_temp[src_col], ran_rotinv_mat)
                color_cats_ref_temp[ref_col] = Transform.transform_single(color_cats_ref_temp[ref_col], ran_rotinv_mat)

                total_scaling = ran_rotinv_mat @ scale_down_mat @ scale_up_mat @ ran_rot_mat @ total_scaling

            # move back to original position
            trans_center_back = np.array([[1.0, 0.0, 0.0, ref_cen[0]],
                                          [0.0, 1.0, 0.0, ref_cen[1]],
                                          [0.0, 0.0, 1.0, ref_cen[2]],
                                          [0.0, 0.0, 0.0, 1.0]])
            
            rep = np.tile(ref_cen, (color_cats_src_temp[src_col].shape[0],1))
            color_cats_src_temp[src_col] += rep
            rep = np.tile(ref_cen, (color_cats_ref_temp[ref_col].shape[0],1))
            color_cats_ref_temp[ref_col] += rep
                                     

            total_transform = trans_center_back @ total_scaling @ trans_center
   

            scal[src_col] = total_transform


        return scal
    

    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def applyTransformation(class_pairs, color_cats_src, rotation_matrix, translation_matrix, scaling_matrix):
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            if color_cats_src[src_col].shape[0] == 0:
                continue

            print(src_col + " " + ref_col)
            M = scaling_matrix[src_col] @ (translation_matrix[src_col] @ rotation_matrix[src_col])
            #M = translation_matrix[src_col] @ rotation_matrix[src_col]
            rep = np.tile(M, (color_cats_src[src_col].shape[0],1,1))

            # extend source by one dimension to get homogenous coordinates
            color_cats_src[src_col] = np.concatenate((color_cats_src[src_col], np.ones((color_cats_src[src_col].shape[0], 1))), axis=1)

            color_cats_src[src_col] = np.einsum('ijk,ik->ij', rep, color_cats_src[src_col])

            #remove last dimension to ger cartesian coordinates
            color_cats_src[src_col] = color_cats_src[src_col][:,:3]

            # print(color_cats_src[src_col].shape)
            # print(M.shape)
            # print(rep.shape)
            # exit()

            
        return color_cats_src
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def transform(points, transform):
        points_out = copy.deepcopy(points)
        tempout = copy.deepcopy(points)
        for c in Transform.color_terms:
            # check if category contains points
            if points_out[c].shape[0] == 0:
                continue

            # transform[c] = np.array([[1.0, 0.0, 0.0, 180.0],
            #                             [0.0, 1.0, 0.0, 0.0],
            #                             [0.0, 0.0, 1.0, 0.0],
            #                             [0.0, 0.0, 0.0, 1.0]])

            rep = np.tile(transform[c], (points_out[c].shape[0],1,1))
            # extend source by one dimension to get homogenous coordinates
            points_out[c] = np.concatenate((points_out[c], np.ones((points_out[c].shape[0], 1))), axis=1)

            # if c == "Green":
            #     print(rep[100,:,:])
            #     print(points_out[c][100,:])

            points_out[c] = np.einsum('ijk,ik->ij', rep, points_out[c])
            # if c == "Green":
            #     print(points_out[c][100,:])
            #     exit()

            #remove last dimension to ger cartesian coordinates
            points_out[c] = points_out[c][:,:3]

            # print(np.min(points_out[c][:,0]-points[c][:,0]))
            # print(np.max(points_out[c][:,0]-points[c][:,0]))
            # print(np.min(points_out[c][:,1]-points[c][:,1]))
            # print(np.max(points_out[c][:,1]-points[c][:,1]))
            # print(np.min(points_out[c][:,2]-points[c][:,2]))
            # print(np.max(points_out[c][:,2]-points[c][:,2]))
            # print("\n")

            #print(points_out[c].shape)
            #print(rep.shape)

        return points_out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def transform_weighted(points, memberships, transform):
        points_temp = copy.deepcopy(points)
        points_out = copy.deepcopy(points)


        for cx in Transform.color_terms:
            # check if category contains points
            if points_temp[cx].shape[0] == 0:
                continue

            #points_temp[cx] = points_temp[cx][0:2]
            #points_out[cx] = points_out[cx][0:2]

            #print(points_temp[cx])
            #print(transform)

            cat_temp = np.zeros_like(points_out[cx])
            #movement = np.zeros_like(points_out[cx])

            # iterate over each transformation matrix (11 in total)
            for c in Transform.color_terms:
                rep = np.tile(transform[c], (points_temp[cx].shape[0],1,1))
                # extend source by one dimension to get homogenous coordinates

                
                temp_points = np.concatenate((points_temp[cx], np.ones((points_temp[cx].shape[0], 1))), axis=1)
                temp_points = np.einsum('ijk,ik->ij', rep, temp_points)
                
                #temp_points = np.concatenate((movement, np.ones((points_temp[cx].shape[0], 1))), axis=1)
                #temp_points = np.einsum('ijk,ik->ij', rep, temp_points)

                #remove last dimension to get cartesian coordinates
                temp_points = temp_points[:,:3]

                # weighting of the result with the membership value
                # membership values for all point within main category cx -> (#points, 1)
                membership_vec = memberships[cx][:,Transform.color_terms_id[c]]

                #membership_vec = membership_vec[0:2]
                #print(membership_vec)

                membership_vec = np.concatenate((np.expand_dims(membership_vec,1),np.expand_dims(membership_vec,1),np.expand_dims(membership_vec,1)), axis=1)

                #print(temp_points)
                cat_temp += temp_points * membership_vec

            
            points_out[cx] = cat_temp
            
            #points_out[cx] += cat_temp
            #print(points_out[cx])

            #print("Done")
            #exit()

        return points_out

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def transform_single(points, transform):
        points_out = copy.deepcopy(points)
        # check if category contains points
        if points_out.shape[0] == 0:
            return points_out

        rep = np.tile(transform, (points_out.shape[0],1,1))
        # extend source by one dimension to get homogenous coordinates
        points_out = np.concatenate((points_out, np.ones((points_out.shape[0], 1))), axis=1)

        points_out = np.einsum('ijk,ik->ij', rep, points_out)

        #remove last dimension to ger cartesian coordinates
        points_out = points_out[:,:3]
        return points_out
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_hue_translation(class_pairs, CV_db, color_cats_db):
        rots = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            
            # if src_vol == 0.0 or ref_vol == 0.0:
            #     rots[src_col] = np.eye(4)
            #     rots[src_col] = np.array([[1.0, 0.0, 0.0, 0.0],
            #                             [0.0, 1.0, 0.0, 0.0],
            #                             [0.0, 0.0, 1.0, 0.0],
            #                             [0.0, 0.0, 0.0, 1.0]])
            #     continue

            src_cen_pol = ColorSpace.cartesian_to_polar(np.expand_dims(src_cen, axis=0))[0,:]
            ref_cen_pol = ColorSpace.cartesian_to_polar(np.expand_dims(ref_cen, axis=0))[0,:]

            db_cen_src = np.sum(color_cats_db[src_col], axis=0) / color_cats_db[src_col].shape[0]
            db_cen_src_pol = ColorSpace.cartesian_to_polar(np.expand_dims(db_cen_src, axis=0))[0,:]

            db_cen_ref = np.sum(color_cats_db[ref_col], axis=0) / color_cats_db[ref_col].shape[0]
            db_cen_ref_pol = ColorSpace.cartesian_to_polar(np.expand_dims(db_cen_ref, axis=0))[0,:]


            if src_vol == 0.0 and ref_vol != 0.0:
                transf = ref_cen_pol - db_cen_src_pol
            elif ref_vol == 0.0 and src_vol != 0.0:
                transf = db_cen_ref_pol - src_cen_pol
            elif src_vol != 0.0 and ref_vol != 0.0:
                transf = ref_cen_pol - src_cen_pol
            else:
                transf = np.zeros(3)

            print(src_col + " - " + ref_col)
            print(transf)

            translation_mat = np.array([[1.0, 0.0, 0.0, transf[0]],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0]])
            
            # translation_mat = np.array([[1.0, 0.0, 0.0, transf[0]],
            #                             [0.0, 1.0, 0.0, 0.0],
            #                             [0.0, 0.0, 1.0, 0.0],
            #                             [0.0, 0.0, 0.0, 1.0]])

            rots[src_col] = translation_mat
        return rots
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_hue_scaling(class_pairs, color_cats_src, color_cats_ref):
        scal = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_cats_src_temp = copy.deepcopy(color_cats_src)
        color_cats_ref_temp = copy.deepcopy(color_cats_ref)

        for elem in class_pairs:
            src_col, src_vol, src_cen, _, _ = elem[0]
            ref_col, ref_vol, ref_cen, _, _ = elem[1]

            if src_vol == 0.0 or ref_vol == 0.0:
                scal[src_col] = np.eye(4)
                continue


            ref_polar_center = ColorSpace.cartesian_to_polar(np.expand_dims(ref_cen, 0))[0]
            ref_polar_center[0] = 0.0


            trans_center = np.array([[1.0, 0.0, 0.0, -ref_polar_center[0]],
                                     [0.0, 1.0, 0.0, -ref_polar_center[1]],
                                     [0.0, 0.0, 1.0, -ref_polar_center[2]],
                                     [0.0, 0.0, 0.0, 1.0        ]])
            


            rep = np.tile(ref_polar_center, (color_cats_src_temp[src_col].shape[0],1))
            color_cats_src_temp[src_col] -= rep
            rep = np.tile(ref_polar_center, (color_cats_ref_temp[ref_col].shape[0],1))
            color_cats_ref_temp[ref_col] -= rep



            _, y_min, z_min = np.amin(color_cats_src_temp[src_col], axis=0)
            _, y_max, z_max = np.amax(color_cats_src_temp[src_col], axis=0)

            y_stretch = y_max - y_min
            z_stretch = z_max - z_min
            scale_down = np.array([1.0, 1.0/y_stretch, 1.0/z_stretch])
            
            scale_down_m = np.tile(scale_down, (color_cats_src_temp[src_col].shape[0],1))
            color_cats_src_temp[src_col] = color_cats_src_temp[src_col] * scale_down_m

            #scale_down_mat = np.eye(4)
            scale_down_mat = np.array([[scale_down[0], 0.0            , 0.0            , 0.0],
                                        [0.0            , scale_down[1], 0.0            , 0.0],
                                        [0.0            , 0.0            , scale_down[2], 0.0],
                                        [0.0            , 0.0            , 0.0            , 1.0]])


            _, y_min, z_min = np.amin(color_cats_ref_temp[ref_col], axis=0)
            _, y_max, z_max = np.amax(color_cats_ref_temp[ref_col], axis=0)

            y_stretch = y_max - y_min
            z_stretch = z_max - z_min
            scale_up = np.array([1.0, y_stretch, z_stretch])

            scale_up_m = np.tile(scale_up, (color_cats_src_temp[src_col].shape[0],1))
            color_cats_src_temp[src_col] = color_cats_src_temp[src_col] * scale_up_m

            scale_up_mat = np.array([[scale_up[0], 0.0          , 0.0          , 0.0],
                                        [0.0          , scale_up[1], 0.0          , 0.0],
                                        [0.0          , 0.0          , scale_up[2], 0.0],
                                        [0.0          , 0.0          , 0.0          , 1.0]])

            # move back to original position
            trans_center_back = np.array([[1.0, 0.0, 0.0, ref_polar_center[0]],
                                          [0.0, 1.0, 0.0, ref_polar_center[1]],
                                          [0.0, 0.0, 1.0, ref_polar_center[2]],
                                          [0.0, 0.0, 0.0, 1.0]])
            
            rep = np.tile(ref_polar_center, (color_cats_src_temp[src_col].shape[0],1))
            color_cats_src_temp[src_col] += rep
            rep = np.tile(ref_polar_center, (color_cats_ref_temp[ref_col].shape[0],1))
            color_cats_ref_temp[ref_col] += rep
                                     

            total_transform = trans_center_back @ scale_down_mat @ scale_up_mat @ trans_center
   

            scal[src_col] = total_transform

        return scal
    