import csv
import numpy as np
from ColorSpace import ColorSpace
from FaissKNeighbors import FaissKNeighbors
import open3d as o3d
from pyhull.convex_hull import ConvexHull
from sklearn.decomposition import PCA
import copy 

class ColorClustering():

    color_terms = np.array(["Red", "Yellow", "Green", "Blue", "Black", "White", "Grey", "Orange", "Brown", "Pink", "Purple"])

    # this variable is only used for rendering, not for the actual algorithm
    color_samples = {
        "Red": np.array([1.0,0.0,0.0]),
        "Yellow":np.array([1.0,1.0,0.0]),
        "Green": np.array([0.0,1.0,0.0]),
        "Blue": np.array([0.0,0.0,1.0]),
        "Black": np.array([0.0,0.0,0.0]),
        "White": np.array([1.0,1.0,1.0]),
        "Grey": np.array([0.5,0.5,0.5]),
        "Orange": np.array([1.0,0.5,0.0]),
        "Brown": np.array([0.4,0.2,0.1]),
        "Pink": np.array([0.85,0.5,0.75]),
        "Purple": np.array([0.4,0.01,0.77]),
    }
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def fuzzy_knn(colors, labels, src_color, ref_color, k=100):
        src_color_copy = copy.deepcopy(src_color)
        ref_color_copy = copy.deepcopy(ref_color)

        src_num = src_color_copy.shape[0]
        ref_num = ref_color_copy.shape[0]

        neigh = FaissKNeighbors(k=k)
        neigh.fit(colors, labels)

        src_preds, src_votes, src_distances = neigh.predict(src_color_copy) 
        ref_preds, ref_votes, ref_distances = neigh.predict(ref_color_copy)

        # shape -> (#points, #labels)
        src_membership = ColorClustering.__calc_membership(src_votes, src_distances, src_num, 2)
        ref_membership = ColorClustering.__calc_membership(ref_votes, ref_distances, ref_num, 2)


        # sort colors by their categories with membership
        # [1] color_cats_src["Red"]: colors which belongs to the category Red
        #     -> shape: (#colors, 3)
        # [2] color_cats_src_ids["Red"]: initial positions within the original color array 
        #     -> shape: (#colors, 1)
        #     -> necessary in order to get the initial image
        # [3] color_cats_ref_mem["Red"]: membership to labels per color
        #     -> shape: (#colors, 11)
        color_cats_src, color_cats_src_ids, color_cats_src_mem = ColorClustering.__sort_by_category(src_preds, src_color_copy, src_membership)
        color_cats_ref, color_cats_ref_ids, color_cats_ref_mem = ColorClustering.__sort_by_category(ref_preds, ref_color_copy, ref_membership)

        return color_cats_src, color_cats_src_ids, color_cats_src_mem, color_cats_ref, color_cats_ref_ids, color_cats_ref_mem
    
    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_colormapping_dataset(path):
        #centers = {"Red":np.zeros(3),"Yellow":np.zeros(3),"Green":np.zeros(3),"Blue":np.zeros(3),"Black":np.zeros(3),"White":np.zeros(3),"Grey":np.zeros(3),"Orange":np.zeros(3),"Brown":np.zeros(3),"Pink":np.zeros(3),"Purple":np.zeros(3)}
        #num_per_cat = {"Red":0,"Yellow":0,"Green":0,"Blue":0,"Black":0,"White":0,"Grey":0,"Orange":0,"Brown":0,"Pink":0,"Purple":0}
        color_clus = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_mapping = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    color_mapping.append([float(row[0]), float(row[1]), float(row[2]), float(np.where(ColorClustering.color_terms == row[3])[0][0])])
                line_count += 1

        color_mapping = np.asarray(color_mapping)
        colors = color_mapping[:,:3] / 255
        colors = np.expand_dims(colors, axis=1).astype("float32")
        hsv_colors = ColorSpace.RGB2cartHSV(colors)
        hsv_colors = np.squeeze(hsv_colors)
        labels = color_mapping[:,3].astype("int64")

        # calculate centers
        for col, lab in zip(hsv_colors, labels):
            color_clus[ColorClustering.color_terms[lab]].append(col)
        #     centers[ColorClustering.color_terms[lab]] += col
        #     num_per_cat[ColorClustering.color_terms[lab]] += 1

        for col in ColorClustering.color_terms:
            color_clus[col] = np.asarray(color_clus[col])

        # for c in ColorClustering.color_terms:
        #     if num_per_cat[c] != 0:
        #         centers[c] = centers[c] / num_per_cat[c]
        #         print(centers[c])

        return hsv_colors, labels, color_clus

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod   
    def __calc_membership(votes, distances, num, m):
        epsilon = 1e-5 # prevents division by 0
        dd = 1.0 / (np.power(distances, 2.0/(m-1.0)) + epsilon)
        denominator = np.sum(dd, axis=1)

        class_num = 11
        membership = np.empty((num, 0))
        for c in range(class_num):
            class_votes = (votes == c).astype(int)
            numerator = np.sum(class_votes * dd, axis=1)
            mem_class = numerator / denominator
            membership = np.concatenate((membership, mem_class[:,np.newaxis]), axis=1)
        return membership

        # neigh = FaissKNeighbors(k=1)
        # neigh.fit(colors, labels)

        # with open('/home/potechius/Downloads/LUT.txt', 'w') as f:
        #     arrays = [np.fromiter(range(256), dtype=int), np.fromiter(range(256), dtype=int), np.fromiter(range(256), dtype=int)]
        #     f.write("red green blue label\n")
        #     for res in itertools.product(*arrays):
        #         print(res)
        #         test_c = cv2.cvtColor(np.asarray(res)[np.newaxis, np.newaxis, :].astype("float32"), cv2.COLOR_RGB2Lab)
        #         src_preds = neigh.predict(test_c[:,0,:])
        #         f.write(str(res[0]) + " " + str(res[1]) + " " + str(res[2]) + " " + str(src_preds[0]) + "\n")
        #         #break

        # exit()

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    @staticmethod   
    def __sort_by_category(predictions, colors, membership):
        color_cats = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_cats_mem = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        color_cats_ids = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}

        for i, (pred, color, mem) in enumerate(zip(predictions, colors, membership)):
            color_cats[ColorClustering.color_terms[int(pred)]].append(color)
            color_cats_mem[ColorClustering.color_terms[int(pred)]].append(mem)
            color_cats_ids[ColorClustering.color_terms[int(pred)]].append(i)

        # converst the color lists to arrays
        for col in ColorClustering.color_terms:
            color_cats[col] = np.asarray(color_cats[col])
            color_cats_mem[col] = np.asarray(color_cats_mem[col])
            color_cats_ids[col] = np.asarray(color_cats_ids[col])

        return color_cats, color_cats_ids, color_cats_mem
    
    # ------------------------------------------------------------------------------------------------------------------
    # Get transfer direction between source and reference
    # Note: White, Grey and Black will be transformed to White, Grey and Black
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_transfer_direction(CV_src, CV_ref, EVV_src, EVV_ref, color_cats_src, color_cats_ref):
        predefined_pairs = [
            #("White", "White"), ("Grey","Grey"), ("Black","Black"), ("Purple", "Pink"), ("Green", "Green"), ("Blue", "Blue")
            #("Green", "Green"), ("Brown","Brown"), ("Red","Red"), ("Black","Black"), ("Grey","Orange"), ("Yellow","Yellow")
        ]

        volumes_src = []
        volumes_ref = []
        for c in ColorClustering.color_terms:
            # (Color, Volume, Center, Eigenvectors, Eigenvalues, #Points)
            volumes_src.append((c, CV_src[c][1], CV_src[c][0], EVV_src[c][0], EVV_src[c][1], color_cats_src[c].shape[0]))
            volumes_ref.append((c, CV_ref[c][1], CV_ref[c][0], EVV_ref[c][0], EVV_ref[c][1], color_cats_ref[c].shape[0]))

        # create class pairs for white-white, grey-grey and black-black
        class_pairs_wgb = []
        #fixed = ["White", "Grey", "Black"]
        for elem_src, elem_ref in predefined_pairs:
            col_src = next(filter(lambda x : x[0]==elem_src, volumes_src))
            col_ref = next(filter(lambda x : x[0]==elem_ref, volumes_ref))
            class_pairs_wgb.append([col_src, col_ref])

        # get transfer direction of similar classes if the volumes have only a 20% difference
        # TODO

        # remove white, grey and black for the sorting procedure
        # col_src = list(filter(lambda x : x[0]!="White" and x[0]!="Grey" and x[0]!="Black", volumes_src))
        # col_ref = list(filter(lambda x : x[0]!="White" and x[0]!="Grey" and x[0]!="Black", volumes_ref))
        col_src = list(filter(lambda x : x[0] not in [a for a, _ in predefined_pairs], volumes_src))
        col_ref = list(filter(lambda x : x[0] not in [b for _, b in predefined_pairs], volumes_ref))

        #sorted_volumes_src = sorted(col_src, key=lambda x: x[1])
        #sorted_volumes_ref = sorted(col_ref, key=lambda x: x[1])
        sorted_volumes_src = sorted(col_src, key=lambda x: x[5])
        sorted_volumes_ref = sorted(col_ref, key=lambda x: x[5])


        # TODO: Number of points will be removed
        class_pairs = [[s[:5], r[:5]] for s, r in zip(sorted_volumes_src,sorted_volumes_ref)]
        return class_pairs_wgb + class_pairs

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    def calc_bary_center_volume(CHs):
        CV = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for c in ColorClustering.color_terms:
            mesh, validity = CHs[c] 
            if not validity:
                b_center_src = (0.0,0.0,0.0)
                vol_src = 0.0
                CV[c] = (b_center_src, vol_src)
                continue

            # calculate gravitational center of convex hull
            # (1) get geometrical center
            coord_center = mesh.get_center()
            #meshw = meshw.translate(-coord_center)
            # (2) iterate over triangles and calculate tetrahaedon mass and center using the coordinate center of the whole mesh
            vol_center = 0
            vertices = np.asarray(mesh.vertices)
            mesh_volume = 0
            for tri in mesh.triangles:
                # calculate center
                pos0 = vertices[tri[0]]
                pos1 = vertices[tri[1]]
                pos2 = vertices[tri[2]]
                pos3 = coord_center
                geo_center = np.sum([pos0, pos1, pos2, pos3], axis=0) / 4
                # calculate volume using the formula:
                # V = |(a-b) * ((b-d) x (c-d))| / 6
                vol = np.abs(np.dot((pos0 - pos3), np.cross((pos1 - pos3), (pos2-pos3)))) / 6
                vol_center += vol * geo_center
                mesh_volume += vol
            # (3) calculate mesh center based on:
            # mass_center = sum(tetra_volumes*tetra_centers)/sum(volumes)

            if mesh_volume <= 0.0:
                CV[c] = ((0.0,0.0,0.0), 0.0)
                CHs[c] = (None, False)
            else:
                mass_center = vol_center / mesh_volume
                CV[c] = (mass_center, mesh_volume)
        return CV
    

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------  
    @staticmethod   
    def calc_convex_hulls(points):
        CH = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for c in ColorClustering.color_terms:
            # Check if array has enough points to create convex hull
            if len(points[c]) <= 4:
                CH[c] = (None, False)
                continue
            # check if array has enough different vectors
            if np.unique(points[c], axis=0).shape[0] <= 4:
                CH[c] = (None, False)
                continue
            
            #try:
            chull_red_src = ConvexHull(points[c])
            #except:
            #    CH[c] = (None, False)
            #    continue

            
            chull_red_src_p = np.expand_dims(chull_red_src.points, axis=1).astype("float32")
            chull_red_src_p = np.squeeze(chull_red_src_p)

            mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(chull_red_src_p),
                                             triangles=o3d.utility.Vector3iVector(chull_red_src.vertices))
            CH[c] = (mesh, True)
        return CH
    
    # ------------------------------------------------------------------------------------------------------------------
    # calculates eigenvectors and -values
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def getEigen(CH):
        EVV = {"Red":[],"Yellow":[],"Green":[],"Blue":[],"Black":[],"White":[],"Grey":[],"Orange":[],"Brown":[],"Pink":[],"Purple":[]}
        for c in ColorClustering.color_terms:
            c_hull, validity = CH[c]
            if validity:
                # Resampling of the convex hulls as uniformly distributed point cloud
                pc_src = c_hull.sample_points_uniformly(number_of_points=1000)
                pc_src.colors = o3d.utility.Vector3dVector(np.full((1000,3), ColorClustering.color_samples[c]))
                # apply PCA to get eigenvectors and -values
                pca_src = PCA(n_components = 3)
                pca_src.fit_transform(np.asarray(pc_src.points))
                eigenvectors_src = pca_src.components_
                eigenvalues_src = pca_src.explained_variance_
            else:
                eigenvectors_src = [(0.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0)]
                eigenvalues_src = [0.0, 0.0, 0.0]
            EVV[c] = (eigenvectors_src, eigenvalues_src)
        return EVV