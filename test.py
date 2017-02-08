import libpointmatcherPythonWrapper
import numpy as np
libpointmatcherPythonWrapper.say_hello("dafdas")
libpointmatcherPythonWrapper.print_ndarray_info(np.random.rand(3, 4).astype(dtype=np.float32))

def do_slc_alignment(self, camera_name, slc_scene_data, template_voxel_dim, scene_voxel_dim, is_force_2D,
                     is_point_to_plane, is_icp_debug):
    print("slc scene contains %d points." % slc_scene_data.shape[0])
    init_translation = np.zeros(3, dtype=np.float32)  # no translation
    # template_to_scene_transformation_matirx = alignWithGicpAndPcl.alignTemplateWithSceneICP(
    template_to_scene_transformation_matirx = alignWithPM.alignTemplateWithSceneICP(
        "what ever path:/home/gao/PycharmProjects/rosEuRocPython/slc_refined_simplified_scaled_0.01x_rotated_90x_180z.ply",
        "what ever path:/home/gao/PycharmProjects/rosEuRocPython/one_slc_on_shelf_scaled_10x.pcd",
        slc_scene_data, init_translation, self.slc_template_point_cloud_3d_npy_array, template_voxel_dim,
        scene_voxel_dim, is_force_2D, is_point_to_plane, is_icp_debug)
    print("template to scene transformation matrix:")
    print(template_to_scene_transformation_matirx)
    print(camera_name + " slc cloud aligned.")
    return template_to_scene_transformation_matirx[:4, :4]
