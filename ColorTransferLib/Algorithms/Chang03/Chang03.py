"""
Copyright 2026 by Herbert Potechius,
Technical University of Berlin
Faculty IV - Electrical Engineering and Computer Science - Institute of Telecommunication Systems - Communication Systems Group
All rights reserved.
This file is released under the "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import numpy as np
from copy import deepcopy
import csv
import open3d as o3d
import time

from ColorTransferLib.Utils.Helper import init_model_files
from pyhull.convex_hull import ConvexHull
from .FaissKNeighbors import FaissKNeighbors
from ColorTransferLib.DataTypes.Video import Video
from ColorTransferLib.DataTypes.VolumetricVideo import VolumetricVideo
from ColorTransferLib.Utils.ColorSpaces import ColorSpaces

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Based on the paper:
#   Title: A framework for transfer colors based on the basic color categories
#   Author: Youngha Chang, Suguru Saito, Masayuki Nakajima
#   Published in: Proceedings Computer Graphics International
#   Year of Publication: 2003
#
# Abstract:
#   Usually, paintings are more appealing than photographic images. This is because paintings have styles. This style 
#   can be distinguished by looking at elements such as motif, color, shape deformation and brush texture. We focus on 
#   the effect of "color" element and devise a method for transforming the color of an input photograph according to a 
#   reference painting. To do this, we consider basic color category concepts in the color transformation process. By 
#   doing so, we achieve large but natural color transformations of an image.
#
# Info:
#   Name: BasicColorCategoryTransfer
#   Identifier: Chang03
#   Link: https://doi.org/10.1109/CGI.2003.1214463
#
# Misc:
#   RayCasting: http://www.open3d.org/docs/latest/tutorial/geometry/ray_casting.html
#
# Implementation Details:
#   The number of colors per category has to be at least 4 with unique position in order to generate a convex hull.
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class Chang03:
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
    # Checks source and reference compatibility
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def apply(src, ref, opt):
        output = {
            "status_code": 0,
            "response": "",
            "object": None,
            "process_time": 0
        }

        if ref.get_type() == "Video" or ref.get_type() == "VolumetricVideo" or ref.get_type() == "LightField":
            output["response"] = "Incompatible reference type."
            output["status_code"] = -1
            return output

        start_time = time.time()

        if src.get_type() == "Image":
            out_obj = Chang03.__apply_image(src, ref, opt)
        elif src.get_type() == "LightField":
            out_obj = Chang03.__apply_lightfield(src, ref, opt)
        elif src.get_type() == "Video":
            out_obj = Chang03.__apply_video(src, ref, opt)
        elif src.get_type() == "VolumetricVideo":
            out_obj = Chang03.__apply_volumetricvideo(src, ref, opt)
        elif src.get_type() == "GaussianSplatting":
            out_obj = Chang03.__apply_gaussiansplatting(src, ref, opt)
        elif src.get_type() == "PointCloud":
            out_obj = Chang03.__apply_pointcloud(src, ref, opt)
        elif src.get_type() == "Mesh":
            out_obj = Chang03.__apply_mesh(src, ref, opt)
        else:
            output["response"] = "Incompatible type."
            output["status_code"] = -1

        output["process_time"] = time.time() - start_time
        output["object"] = out_obj

        return output

    # ------------------------------------------------------------------------------------------------------------------
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __color_transfer(src_color, ref_color, opt):
        model_file_paths = init_model_files("Chang03", ["colormapping.csv"])

        # ---------------------------------------------------------------------
        # 1) Convert source / reference to working colorspace
        #    src_color, ref_color: (N,1,3) in RGB [0,1]
        # ---------------------------------------------------------------------
        src_arr = np.asarray(src_color, dtype=np.float32)
        ref_arr = np.asarray(ref_color, dtype=np.float32)

        if src_arr.shape[-1] != 3 or ref_arr.shape[-1] != 3:
            raise ValueError(
                f"__color_transfer expects last dimension = 3, "
                f"got src_shape={src_arr.shape}, ref_shape={ref_arr.shape}"
            )

        if opt.colorspace == "cielab":
            # OpenCV CIELAB
            flat_src = src_arr.reshape(-1, 3)
            flat_ref = ref_arr.reshape(-1, 3)
            src_cs = ColorSpaces.rgb_to_cielab(flat_src).reshape(src_arr.shape)
            ref_cs = ColorSpaces.rgb_to_cielab(flat_ref).reshape(ref_arr.shape)
        elif opt.colorspace == "lalphabeta":
            # Ruderman lαβ
            flat_src = src_arr.reshape(-1, 3)
            flat_ref = ref_arr.reshape(-1, 3)
            src_cs = ColorSpaces.rgb_to_lalphabeta(flat_src).reshape(src_arr.shape)
            ref_cs = ColorSpaces.rgb_to_lalphabeta(flat_ref).reshape(ref_arr.shape)
        elif opt.colorspace == "rgb":
            # Work directly in RGB
            src_cs = src_arr
            ref_cs = ref_arr
        else:
            raise ValueError(f"Unsupported colorspace: {opt.colorspace}")

        # ---------------------------------------------------------------------
        # 2) Load BCC color mapping and convert to same working colorspace
        # ---------------------------------------------------------------------
        color_terms = np.array([
            "Red", "Yellow", "Green", "Blue", "Black",
            "White", "Grey", "Orange", "Brown", "Pink", "Purple"
        ])
        color_mapping = []
        with open(model_file_paths["colormapping.csv"]) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    color_mapping.append([
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(np.where(color_terms == row[3])[0][0])
                    ])
                line_count += 1

        color_mapping = np.asarray(color_mapping, dtype=np.float32)

        # ------------------------------------------------------------------
        # handle empty or invalid color_mapping gracefully
        # fall back to identity mapping (no category-based transfer)
        # ------------------------------------------------------------------
        if color_mapping.size == 0:
            # src_cs has shape (N,1,3) in working colours
            sorted_colors = src_cs[:, 0, :].reshape(-1, 3)

            if opt.colorspace == "cielab":
                out_rgb = ColorSpaces.cielab_to_rgb(sorted_colors)
            elif opt.colorspace == "lalphabeta":
                out_rgb = ColorSpaces.lalphabeta_to_rgb(sorted_colors)
            else:  # "rgb"
                out_rgb = sorted_colors

            out_rgb = np.clip(out_rgb, 0.0, 1.0).astype(np.float32)
            return out_rgb

        # ensure 2D shape (M,4) even if only one row
        color_mapping = color_mapping.reshape(-1, 4)

        colors_rgb = color_mapping[:, :3] / 255.0  # (M,3) in RGB [0,1]
        labels = color_mapping[:, 3].astype("int64")

        if opt.colorspace == "cielab":
            colors_cs = ColorSpaces.rgb_to_cielab(colors_rgb)
        elif opt.colorspace == "lalphabeta":
            colors_cs = ColorSpaces.rgb_to_lalphabeta(colors_rgb)
        elif opt.colorspace == "rgb":
            colors_cs = colors_rgb
        else:
            raise ValueError(f"Unsupported colorspace: {opt.colorspace}")

        # ---------------------------------------------------------------------
        # 3) Train classifier in working colorspace
        # ---------------------------------------------------------------------
        neigh = FaissKNeighbors(k=1)
        neigh.fit(colors_cs.astype("float32"), labels)

        # predict src / ref labels (colors are of size (N,1,3))
        src_preds = neigh.predict(src_cs[:, 0, :])  # (N,)
        ref_preds = neigh.predict(ref_cs[:, 0, :])

        # ---------------------------------------------------------------------
        # 4) Build per-category color lists
        # ---------------------------------------------------------------------
        color_cats_src = {name: [] for name in color_terms}
        color_cats_src_ids = {name: [] for name in color_terms}
        color_cats_ref = {name: [] for name in color_terms}

        for i, (pred, color) in enumerate(zip(src_preds, src_cs[:, 0, :])):
            key = color_terms[int(pred)]
            color_cats_src[key].append(color)
            color_cats_src_ids[key].append(i)

        for pred, color in zip(ref_preds, ref_cs[:, 0, :]):
            key = color_terms[int(pred)]
            color_cats_ref[key].append(color)

        # ---------------------------------------------------------------------
        # 5) Per-category convex hull + radial transfer
        # ---------------------------------------------------------------------
        output_colors = np.empty((0, 3), dtype=np.float32)
        output_ids = np.empty((0, 1), dtype=np.int64)

        for color_cat in color_cats_src.keys():
            src_list = color_cats_src[color_cat]
            ref_list = color_cats_ref[color_cat]

            if len(color_cats_src_ids[color_cat]) == 0:
                continue

            output_ids = np.concatenate(
                (output_ids, np.asarray(color_cats_src_ids[color_cat], dtype=np.int64)[:, np.newaxis]),
                axis=0,
            )

            # not enough reference colors -> no transfer for this category
            if len(src_list) >= 4 and len(ref_list) < 4:
                output_colors = np.concatenate(
                    (output_colors, np.asarray(src_list, dtype=np.float32)),
                    axis=0,
                )
                continue
            elif len(src_list) == 0:
                continue
            elif len(src_list) < 4:
                output_colors = np.concatenate(
                    (output_colors, np.asarray(src_list, dtype=np.float32)),
                    axis=0,
                )
                continue

            src_arr_cat = np.asarray(src_list, dtype=np.float32)
            ref_arr_cat = np.asarray(ref_list, dtype=np.float32)

            # --- Degeneracy checks: too few unique points OR (near-)coplanar ---
            if (
                Chang03.__check_identity(src_arr_cat)
                or Chang03.__check_identity(ref_arr_cat)
                or Chang03.__check_coplanarity(src_arr_cat)
                or Chang03.__check_coplanarity(ref_arr_cat)
            ):
                # fallback: keep original source colors for this category
                output_colors = np.concatenate((output_colors, src_arr_cat), axis=0)
                continue

            mesh_src = Chang03.__calc_convex_hull(src_arr_cat)
            mesh_ref = Chang03.__calc_convex_hull(ref_arr_cat)

            mass_center_src = Chang03.__calc_gravitational_center(mesh_src)
            mass_center_ref = Chang03.__calc_gravitational_center(mesh_ref)

            # intersections along rays from mass center
            inter_src = Chang03.__calc_line_mesh_intersection(
                mesh_src, src_arr_cat - mass_center_src, mass_center_src
            )
            inter_ref = Chang03.__calc_line_mesh_intersection(
                mesh_ref, src_arr_cat - mass_center_src, mass_center_ref
            )

            # Color Transfer (in working colorspace)
            output_colors = Chang03.__transfer_colors(
                output_colors=output_colors,
                colors=src_arr_cat,
                mass_center_src=mass_center_src,
                mass_center_ref=mass_center_ref,
                dist_src=inter_src["t_hit"],
                dist_ref=inter_ref["t_hit"],
            )

        # ---------------------------------------------------------------------
        # 6) Reorder to original pixel order
        # ---------------------------------------------------------------------
        sort_idx = np.argsort(output_ids, axis=0).reshape(-1)
        sorted_colors = output_colors[sort_idx]

        # ---------------------------------------------------------------------
        # 7) Convert back to RGB depending on working colorspace
        # ---------------------------------------------------------------------
        if opt.colorspace == "cielab":
            out_rgb = ColorSpaces.cielab_to_rgb(sorted_colors)

        elif opt.colorspace == "lalphabeta":
            out_rgb = ColorSpaces.lalphabeta_to_rgb(sorted_colors)
        else:  # "rgb"
            out_rgb = sorted_colors

        out_rgb = np.clip(out_rgb, 0.0, 1.0).astype(np.float32)
        return out_rgb

    # ------------------------------------------------------------------------------------------------------------------
    # checks if the given data does not lie on a plane -> this would lead to a convex hull with volume = 0
    # returns True if points are (near-)coplanar (degenerate for 3D convex hull)
    # ------------------------------------------------------------------------------------------------------------------  
    def __check_coplanarity(data, tol: float = 1e-6):
        points = np.asarray(data, dtype=np.float32)

        # fewer than 4 points cannot form a 3D volume
        if points.shape[0] < 4:
            return True

        # center points
        centered = points - points.mean(axis=0, keepdims=True)

        # SVD to check rank / dimensionality
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            # if SVD fails, treat as degenerate
            return True

        # if smallest singular value is much smaller than largest -> effectively planar
        if s[0] <= 0:
            return True  # all zeros or severely degenerate

        ratio = s[-1] / s[0]
        return ratio < tol

    # ------------------------------------------------------------------------------------------------------------------
    # checks if the given data contains at least four different values for creating a convex hull with volume > 0
    # returns true if the data does not contain at least four different values
    # ------------------------------------------------------------------------------------------------------------------  
    def __check_identity(data):
        unique_val = np.unique(data, axis=0)
        if unique_val.shape[0] < 4:
            return True
        else:
            return False

    # ------------------------------------------------------------------------------------------------------------------
    # Calculates the gravitational center of a mesh
    # ------------------------------------------------------------------------------------------------------------------  
    def __calc_gravitational_center(mesh):
        # calculate gravitational center of convex hull
        # (1) get geometrical center
        coord_center = mesh.get_center()
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
            # calculate volume using the formula: V = |(a-b) * ((b-d) x (c-d))| / 6
            vol = np.abs(np.dot((pos0 - pos3), np.cross((pos1 - pos3), (pos2-pos3)))) / 6
            vol_center += vol * geo_center
            mesh_volume += vol
        # (3) calculate mesh center based on: mass_center = sum(tetra_volumes*tetra_centers)/sum(volumes)
        mass_center = vol_center / mesh_volume
        return mass_center
    
    # ------------------------------------------------------------------------------------------------------------------
    # Calculates the convex hull of a given point set
    # ------------------------------------------------------------------------------------------------------------------  
    def __calc_convex_hull(points):
        chull_red_src = ConvexHull(points)
        chull_red_src_p = np.expand_dims(chull_red_src.points, axis=1).astype("float32")
        chull_red_src_p = np.squeeze(chull_red_src_p)

        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(chull_red_src_p),
                                          triangles=o3d.utility.Vector3iVector(chull_red_src.vertices))
        return mesh
    
    # ------------------------------------------------------------------------------------------------------------------
    # Calculates the intersection between a line and a triangle mesh
    # ------------------------------------------------------------------------------------------------------------------    
    def __calc_line_mesh_intersection(mesh, directions, mass_center):
        scene = o3d.t.geometry.RaycastingScene()
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh_id = scene.add_triangles(mesh)

        # Note: directions have to be normalized in order to get the correct ray cast distance
        norms = np.linalg.norm(directions, axis=1)[:, np.newaxis]
        norms_ext = np.concatenate((norms, norms, norms), axis= 1)
        norm_directions = directions / norms_ext

        rays_src = np.concatenate((np.full(np.asarray(directions).shape, mass_center), norm_directions), axis=1)

        rays_src_tensor = o3d.core.Tensor(rays_src, dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays_src_tensor)
        return ans
    
    # ------------------------------------------------------------------------------------------------------------------
    # Calculates the convex hull of a given point set and saves it as a triangle mesh
    # ------------------------------------------------------------------------------------------------------------------ 
    def __write_convex_hull_mesh(colors, shape, path, color, color_space="LAB"):
        if color_space == "RGB":
            ex = np.asarray(colors)[:, np.newaxis]
            cex = ColorSpaces.cielab_to_rgb(ex)
            
            mesh = Chang03.__calc_convex_hull(cex.squeeze())
        else:
            mesh = Chang03.__calc_convex_hull(colors)

        colors = np.full(shape, color)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_triangle_mesh(filename=path, 
                                   mesh=mesh, 
                                   write_ascii=True,
                                   write_vertex_normals=False,
                                   write_vertex_colors=True,
                                   write_triangle_uvs=False)
        
    # ------------------------------------------------------------------------------------------------------------------
    # coplanarity
    # ------------------------------------------------------------------------------------------------------------------         
    def __transfer_colors(output_colors, colors, mass_center_src, mass_center_ref, dist_src, dist_ref):
        """
        Robust radial transfer inside a color category.
        Handles degenerate rays / missing intersections by leaving colors untouched.
        """
        colors = np.asarray(colors, dtype=np.float32)  # (N,3)
        point_dir = colors - mass_center_src  # (N,3)
        point_dist = np.linalg.norm(point_dir, axis=1)  # (N,)

        dist_src_np = dist_src.numpy()  # (N,)
        dist_ref_np = dist_ref.numpy()  # (N,)

        eps = 1e-6
        big = 1e6

        valid = (
            np.isfinite(point_dist)
            & np.isfinite(dist_src_np)
            & np.isfinite(dist_ref_np)
            & (point_dist > eps)
            & (dist_src_np > eps)
            & (dist_src_np < big)
            & (dist_ref_np > eps)
            & (dist_ref_np < big)
        )

        # start with original colors; only overwrite valid ones
        out = colors.copy()

        if np.any(valid):
            pd = point_dist[valid][:, np.newaxis]  # (M,1)
            norm_point_dir = point_dir[valid] / pd  # (M,3)

            rel = (point_dist[valid] / dist_src_np[valid])[:, np.newaxis]  # (M,1)
            ref_d = dist_ref_np[valid][:, np.newaxis]  # (M,1)

            shift = norm_point_dir * ref_d * rel  # (M,3)
            out[valid] = shift + mass_center_ref  # broadcast mass_center_ref (3,)

        output_colors = np.concatenate((output_colors, out), axis=0)
        return output_colors
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_image(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = Chang03.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_video(src, ref, opt): 
        # check if type is video
        out_colors_arr = []
        src_colors = src.get_colors()

        for i, src_color in enumerate(src_colors):
            # Preprocessing
            ref_color = ref.get_colors()
            out_img = deepcopy(src.get_images()[0])

            out_colors = Chang03.__color_transfer(src_color, ref_color, opt)

            out_img.set_colors(out_colors)
            out_colors_arr.append(out_img)

        outp = Video(imgs=out_colors_arr, fps=src.get_fps())

        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_volumetricvideo(src, ref, opt): 
        out_colors_arr = []
        src_colors = src.get_colors()

        for i, src_color in enumerate(src_colors):
            # Preprocessing
            ref_color = ref.get_colors()
            out_img = deepcopy(src.get_meshes()[i])

            out_colors = Chang03.__color_transfer(src_color, ref_color, opt)

            out_img.set_colors(out_colors)
            out_colors_arr.append(out_img)
            outp = VolumetricVideo(meshes=out_colors_arr, file_name=src.get_file_name())

        return outp

    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_lightfield(src, ref, opt):
        src_lightfield_array = src.get_image_array()
        out = deepcopy(src)
        out_lightfield_array = out.get_image_array()

        for row in range(src.get_grid_size()[0]):
            for col in range(src.get_grid_size()[1]):
                src_color = src_lightfield_array[row][col].get_colors()
                ref_color = ref.get_colors()

                out_colors = Chang03.__color_transfer(src_color, ref_color, opt)

                out_lightfield_array[row][col].set_colors(out_colors)

        return out
    
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_gaussiansplatting(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = Chang03.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_pointcloud(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = Chang03.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp
    # ------------------------------------------------------------------------------------------------------------------
    # Applies the color transfer algorihtm
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def __apply_mesh(src, ref, opt):
        src_color = src.get_colors()
        ref_color = ref.get_colors()
        out_img = deepcopy(src)

        out_colors = Chang03.__color_transfer(src_color, ref_color, opt)

        out_img.set_colors(out_colors)
        outp = out_img
        return outp

