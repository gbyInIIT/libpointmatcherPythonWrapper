#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pcl
import libpointmatcherPythonWrapper
from tf import transformations
import math


# def do_slc_alignment(slc_template_point_cloud_3d_npy_array, slc_scene_point_cloud_3d_npy_array, template_voxel_dim, scene_voxel_dim, is_force_2D, is_point_to_plane, is_icp_debug):
def do_slc_alignment(slc_template_point_cloud_3d_npy_array, slc_scene_point_cloud_3d_npy_array):
    template_voxel_dim = 0.006
    scene_voxel_dim = 0.006
    is_force_2D = 0
    is_point_to_plane = 1
    is_icp_debug = 1
    # print("slc scene contains %d points." % slc_scene_point_cloud_3d_npy_array.shape[0])
    init_translation = np.zeros(3, dtype=np.float32)  # no translation
    template_to_scene_transformation_matirx_5x4 = libpointmatcherPythonWrapper.alignTemplateWithSceneICP(
        "what ever path:/home/gao/PycharmProjects/rosEuRocPython/slc_refined_simplified_scaled_0.01x_rotated_90x_180z.ply",
        "what ever path:/home/gao/PycharmProjects/rosEuRocPython/one_slc_on_shelf_scaled_10x.pcd",
        slc_scene_point_cloud_3d_npy_array, init_translation, slc_template_point_cloud_3d_npy_array, template_voxel_dim,
        scene_voxel_dim, is_force_2D, is_point_to_plane, is_icp_debug)
    # print("template to scene transformation matrix:")
    # print(template_to_scene_transformation_matirx)
    return template_to_scene_transformation_matirx_5x4

def main():
    slc_template_pcl_cloud = pcl.PointCloud()
    # p.from_file("/home/gao/PycharmProjects/rosEuRocPython/slc_refined_simplified_scaled_0.01x_rotated_90x_180z.pcd")
    # p.from_file("/home/gao/PycharmProjects/rosEuRocPython/slc_scaled_rot_from_simulator_stl_rotated_180z_scaled_10x_simplified_edged.pcd")
    # slc_pcl_cloud.from_file('slc_refined_simplified_scaled_1x_rotated_90x_180z.pcd')
    slc_template_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/model_cloud.pcd")
    # slc_template_pcl_cloud.from_file("/home/gao/PycharmProjects/rosEuRocPython/slc_scaled_rot_from_simulator_stl_rotated_180z_scaled_1x_simplified_edged.pcd")
    slc_template_point_cloud_npy_array_nx3 = slc_template_pcl_cloud.to_array()
    scene_pcl_cloud = pcl.PointCloud()
    # nut_pcl_cloud.from_file("nut_scaled_rot_from_simulator_scaled_0.15x.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/template_36_orientations/templates18.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/synthetized_dataset/low perturbation/dataset_0n9.pcd")
    scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/synthetized_dataset/high perturbation/dataset_2n9.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/real_dataset/dataset_-90_c0.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/real_dataset/dataset_+90_c0.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/real_dataset/dataset_180_c0.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/real_dataset/dataset_0_c0.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/real_dataset/dataset_ccSlc1_2c1.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/real_dataset/dataset_ccSlc1_1c0.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/real_dataset/dataset_ccSlc1_1c1.pcd")
    # scene_pcl_cloud.from_file("/home/gao/Downloads/gby_ft_publication/real_dataset/dataset_ccSlc1_2c0.pcd")

    init_rotation_matrix_4x4 = transformations.euler_matrix(0, 0, 0, 'rzyx')
    # init_rotation_matrix_4x4 = transformations.euler_matrix(math.pi, 0, 0, 'rzyx') * init_rotation_matrix_4x4
    slc_scene_point_cloud_npy_array_nx3 = scene_pcl_cloud.to_array()
    n_rotation = 72
    f = open("alignment_error_syn_with_new_lib.csv", 'w')
    print("rotation angles diff, overlap, averagedMatchingDist2, weightedMatchingDist2, nMatch, corrected rotation diff", file=f)
    for i in range(n_rotation):
        rotation_angles = 2 * math.pi/n_rotation*i
        rotation_matrix_4x4 = transformations.euler_matrix(rotation_angles, 0, 0, 'rzyx')
        rotation_matrix_3x3 = np.dot(rotation_matrix_4x4[:3, :3], init_rotation_matrix_4x4[:3, :3])
        rotated_scene_point_cloud_npy_array_nx3 = np.dot(slc_scene_point_cloud_npy_array_nx3, np.transpose(rotation_matrix_3x3)).astype(np.float32)
        aligned_transformation_matrix_5x4 = do_slc_alignment(slc_template_point_cloud_npy_array_nx3, rotated_scene_point_cloud_npy_array_nx3)
        a, b, c = transformations.euler_from_matrix(aligned_transformation_matrix_5x4[:4, :4], 'rzyx')
        overLap = aligned_transformation_matrix_5x4[4, 0]
        averagedMatchingDist2 = aligned_transformation_matrix_5x4[4, 1]
        weightedMatchingDist2 = aligned_transformation_matrix_5x4[4, 2]
        nMatch = aligned_transformation_matrix_5x4[4, 3]
        aligned_rotation_angles = a
        print("rotation angles: %f" % rotation_angles)
        print("rotation angles aligned: %f, overlap: %f" % (aligned_rotation_angles, overLap))
        # print("rotation angles diff: %f" % np.fmod(abs(aligned_rotation_angles-rotation_angles), 2*math.pi))
        print("%f,%f,%e,%e,%f" % (np.fmod(rotation_angles - aligned_rotation_angles, 2*math.pi),
                                     overLap, averagedMatchingDist2, weightedMatchingDist2, nMatch), file=f)
    f.close()


if __name__ == '__main__':
    main()
    # run_dlr()

