#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_common.h>
#include <math.h>
//#include <omp.h>
#include <cblas.h>
#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

// program options
#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/tokenizer.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>   // TicToc

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include "pointmatcher/PointMatcher.h"

#define __IN
#define __OUT

using namespace std;

typedef pcl::PointXYZ PointXYZT;
typedef pcl::PointCloud<PointXYZT> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

DP * pcl_cloud_to_pm_DataPoints_2D_data(PointCloudXYZ::Ptr pcl_cloud) {
    DP::Labels featureLabels;
    featureLabels.push_back(DP::Label("x", 1));
    featureLabels.push_back(DP::Label("y", 1));
    featureLabels.push_back(DP::Label("w", 1));
    DP::Labels descriptorLabels;
    descriptorLabels.push_back(DP::Label("curvatures", 1));
    size_t n_point = pcl_cloud->size();
    PM::Matrix features(3, n_point);
    PM::Matrix descriptors(1, n_point);
    int i_point = 0;
    for (pcl::PointCloud<pcl::PointXYZ>::iterator it = pcl_cloud->begin(); it != pcl_cloud->end(); it++) {
        pcl::PointXYZ pt_pcl = *it;
        features(0, i_point) = pt_pcl.x;
        features(1, i_point) = pt_pcl.y;
        features(2, i_point) = 1.f;
        descriptors(0, i_point) = 1.f;
        i_point++;
    }
//    DP * pm_data_points_ptr = new PM::DataPoints(features, featureLabels);
    DP * pm_data_points_ptr = new PM::DataPoints(features, featureLabels, descriptors, descriptorLabels);
    return pm_data_points_ptr;
}

boost::shared_ptr<DP> pcl_cloud_to_pm_DataPoints(PointCloudXYZ::Ptr pcl_cloud) {
    DP::Labels featureLabels;
    featureLabels.push_back(DP::Label("x", 1));
    featureLabels.push_back(DP::Label("y", 1));
    featureLabels.push_back(DP::Label("z", 1));
    featureLabels.push_back(DP::Label("w", 1));
    DP::Labels descriptorLabels;
    descriptorLabels.push_back(DP::Label("curvatures", 1));
    size_t n_point = pcl_cloud->size();
    PM::Matrix features(4, n_point);
    PM::Matrix descriptors(1, n_point);
    int i_point = 0;
    for (pcl::PointCloud<pcl::PointXYZ>::iterator it = pcl_cloud->begin(); it != pcl_cloud->end(); it++) {
        pcl::PointXYZ pt_pcl = *it;
        features(0, i_point) = pt_pcl.x;
        features(1, i_point) = pt_pcl.y;
        features(2, i_point) = pt_pcl.z;
        features(3, i_point) = 1.f;
        descriptors(0, i_point) = 1.f;
        i_point++;
    }
    boost::shared_ptr<DP> pm_data_points_ptr(new PM::DataPoints(features, featureLabels, descriptors, descriptorLabels));
    return pm_data_points_ptr;
}

PointCloudXYZ::Ptr pm_DataPoints_to_pcl_cloud(const DP & pm_data_points) {
    PointCloudXYZ::Ptr pcl_cloud(new PointCloudXYZ);  // Original point cloud
    unsigned n_point = pm_data_points.getNbPoints();
    const PM::Matrix & features = pm_data_points.features;
    for (unsigned i = 0; i < n_point; i++) {
        pcl::PointXYZ pcl_pt(features(0, i), features(1, i), features(2, i));
        pcl_cloud->push_back(pcl_pt);
    }
    return pcl_cloud;
}

void print4x4Matrix(const Eigen::Matrix4d &matrix) {
    printf("Rotation matrix :\n");
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
    printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
    printf("Translation vector :\n");
    printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

void
keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                      void *nothing) {
    if (event.getKeySym() == "space" && event.keyDown())
        bool debug = true;
}

PointCloudXYZ::Ptr pipeline_cloud_remove_out_liers(PointCloudXYZ::Ptr cloud) {
    PointCloudXYZ::Ptr cloud_filtered (new PointCloudXYZ);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (150);
    sor.setStddevMulThresh (.30);
    sor.filter (*cloud_filtered);
    return cloud_filtered;
}

PointCloudXYZ::Ptr pipeline_cloud_down_sampling(PointCloudXYZ::Ptr cloud, const float voxel_grid_size = 0.05f) {
    // ... and downsampling the point scene_cloud
    pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
    vox_grid.setInputCloud(cloud);
    vox_grid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
    //vox_grid.filter (*scene_cloud); // Please see this http://www.pcl-developers.org/Possible-problem-in-new-VoxelGrid-implementation-from-PCL-1-5-0-td5490361.html
    pcl::PointCloud<pcl::PointXYZ>::Ptr tempCloud(new pcl::PointCloud<pcl::PointXYZ>);
    vox_grid.filter(*tempCloud);
    return tempCloud;
}

void config_icp_general(PM::ICP & icp, int isForce2D = 0, int isPointToPlane = 1) {
    PointMatcherSupport::Parametrizable::Parameters params;
    setLogger(PM::get().LoggerRegistrar.create("FileLogger"));

    PM::Transformation* rigidTransformation = PM::get().TransformationRegistrar.create("RigidTransformation");
    icp.transformations.push_back(rigidTransformation);

    PM::DataPointsFilter* identityDataPointsFilter = PM::get().DataPointsFilterRegistrar.create("IdentityDataPointsFilter");
    //PM::DataPointsFilter* randomSamplingDataPointsFilter = PM::get().DataPointsFilterRegistrar.create("RandomSamplingDataPointsFilter");
    //PM::DataPointsFilter* samplingSurfaceNormalDataPointsFilter = PM::get().DataPointsFilterRegistrar.create("SamplingSurfaceNormalDataPointsFilter");
    PM::DataPointsFilter* surfaceNormalDataPointsFilter = PM::get().DataPointsFilterRegistrar.create("SurfaceNormalDataPointsFilter");
    if (isPointToPlane == 1) {
        if (isForce2D == 1) {
            params["force2D"] = "1";
        } else {
            params["force2D"] = "0";
        }
        PM::ErrorMinimizer * pointToPlaneErrorMinimizer = PM::get().ErrorMinimizerRegistrar.create("PointToPlaneErrorMinimizer", params);
//        PM::ErrorMinimizer * pointToPlaneErrorMinimizer2DRotation = PM::get().ErrorMinimizerRegistrar.create("PointToPlaneErrorMinimizer2DRotation", params);
        icp.errorMinimizer.reset(pointToPlaneErrorMinimizer);
        params.clear();
    } else {
        PM::ErrorMinimizer * pointToPointErrorMinimizer = PM::get().ErrorMinimizerRegistrar.create("PointToPointErrorMinimizer");
//        PM::ErrorMinimizer * pointToPointErrorMinimizer2DRotation = PM::get().ErrorMinimizerRegistrar.create("PointToPointErrorMinimizer2DRotation");
        if (isForce2D == 1) {
            icp.errorMinimizer.reset(pointToPointErrorMinimizer);
        } else {
            icp.errorMinimizer.reset(pointToPointErrorMinimizer);
        }
    }
//    icp.errorMinimizer.reset(pointToPlaneErrorMinimizer2DRotation);

    icp.readingDataPointsFilters.push_back(identityDataPointsFilter);
    icp.referenceDataPointsFilters.push_back(surfaceNormalDataPointsFilter);
//    icp.referenceDataPointsFilters.push_back(identityDataPointsFilter);

    PM::OutlierFilter* trimmedDistOutlierFilter = PM::get().OutlierFilterRegistrar.create("TrimmedDistOutlierFilter");
    icp.outlierFilters.push_back(trimmedDistOutlierFilter);
    PM::OutlierFilter* curvatureOutlierFilter = PM::get().OutlierFilterRegistrar.create("CurvatureOutlierFilter");
    icp.outlierFilters.push_back(curvatureOutlierFilter);

    PM::Matcher* kDTreeMatcher = PM::get().MatcherRegistrar.create("KDTreeMatcher");
    icp.matcher.reset(kDTreeMatcher);


    params["maxIterationCount"] = "5000";
    PM::TransformationChecker* counterTransformationChecker = PM::get().TransformationCheckerRegistrar.create("CounterTransformationChecker", params);
    icp.transformationCheckers.push_back(counterTransformationChecker);
    params.clear();
//    return boost::assign::list_of<ParameterDoc>
//				( "minDiffRotErr", "threshold for rotation error (radian)", "0.001", "0.", "6.2831854", &P::Comp<T> )
//				( "minDiffTransErr", "threshold for translation error", "0.001", "0.", "inf", &P::Comp<T> )
//				( "smoothLength", "number of iterations over which to average the differencial error", "3", "0", "2147483647", &P::Comp<unsigned> )
//			;
    params["minDiffRotErr"] = "0.00005";
    params["minDiffTransErr"] = "0.0005";
    PM::TransformationChecker* differentialTransformationChecker = PM::get().TransformationCheckerRegistrar.create("DifferentialTransformationChecker", params);
    icp.transformationCheckers.push_back(differentialTransformationChecker);
    params.clear();

    PM::Inspector* nullInspector = PM::get().InspectorRegistrar.create("NullInspector");
//    				( "baseFileName", "base file name for the statistics files (if empty, disabled)", "" )
//				( "dumpPerfOnExit", "dump performance statistics to stderr on exit", "0" )
//				( "dumpStats", "dump the statistics on first and last step", "0" )
    params["baseFileName"] = "performace_stat";
    params["dumpPerfOnExit"] = "1";
    params["dumpStats"] = "1";
    //PM::Inspector* performaceInspector = PM::get().InspectorRegistrar.create("PerformanceInspector", params);
    icp.inspector.reset(nullInspector);
//    icp.inspector.reset(performaceInspector);
    params.clear();
    printf("ICP generally configured.\n");
}
void config_icp(PM::ICP & icp, int isForce2D = 0, int isPointToPlane = 1) {
    PointMatcherSupport::Parametrizable::Parameters params;
    setLogger(PM::get().LoggerRegistrar.create("FileLogger"));

    PM::Transformation* rigidTransformation = PM::get().TransformationRegistrar.create("RigidTransformation");
    icp.transformations.push_back(rigidTransformation);

    PM::DataPointsFilter* identityDataPointsFilter = PM::get().DataPointsFilterRegistrar.create("IdentityDataPointsFilter");
    PM::DataPointsFilter* randomSamplingDataPointsFilter = PM::get().DataPointsFilterRegistrar.create("RandomSamplingDataPointsFilter");
    PM::DataPointsFilter* samplingSurfaceNormalDataPointsFilter = PM::get().DataPointsFilterRegistrar.create("SamplingSurfaceNormalDataPointsFilter");
    PM::DataPointsFilter* surfaceNormalDataPointsFilter = PM::get().DataPointsFilterRegistrar.create("SurfaceNormalDataPointsFilter");
    if (isPointToPlane == 1) {
        if (isForce2D == 1) {
            params["force2D"] = "1";
        } else {
            params["force2D"] = "0";
        }
        PM::ErrorMinimizer * pointToPlaneErrorMinimizer = PM::get().ErrorMinimizerRegistrar.create("PointToPlaneErrorMinimizer", params);
        PM::ErrorMinimizer * pointToPlaneErrorMinimizer2DRotation = PM::get().ErrorMinimizerRegistrar.create("PointToPlaneErrorMinimizer2DRotation", params);
        icp.errorMinimizer.reset(pointToPlaneErrorMinimizer2DRotation);
        params.clear();
    } else {
        PM::ErrorMinimizer * pointToPointErrorMinimizer = PM::get().ErrorMinimizerRegistrar.create("PointToPointErrorMinimizer");
        PM::ErrorMinimizer * pointToPointErrorMinimizer2DRotation = PM::get().ErrorMinimizerRegistrar.create("PointToPointErrorMinimizer2DRotation");
        if (isForce2D == 1) {
            icp.errorMinimizer.reset(pointToPointErrorMinimizer2DRotation);
        } else {
            icp.errorMinimizer.reset(pointToPointErrorMinimizer);
        }
    }
//    icp.errorMinimizer.reset(pointToPlaneErrorMinimizer2DRotation);

    icp.readingDataPointsFilters.push_back(identityDataPointsFilter);
    icp.referenceDataPointsFilters.push_back(surfaceNormalDataPointsFilter);

    PM::OutlierFilter* trimmedDistOutlierFilter = PM::get().OutlierFilterRegistrar.create("TrimmedDistOutlierFilter");
    icp.outlierFilters.push_back(trimmedDistOutlierFilter);
    PM::OutlierFilter* curvatureOutlierFilter = PM::get().OutlierFilterRegistrar.create("CurvatureOutlierFilter");
    icp.outlierFilters.push_back(curvatureOutlierFilter);

    PM::Matcher* kDTreeMatcher = PM::get().MatcherRegistrar.create("KDTreeMatcher");
    icp.matcher.reset(kDTreeMatcher);


    params["maxIterationCount"] = "5000";
    PM::TransformationChecker* counterTransformationChecker = PM::get().TransformationCheckerRegistrar.create("CounterTransformationChecker", params);
    icp.transformationCheckers.push_back(counterTransformationChecker);
    params.clear();
//    return boost::assign::list_of<ParameterDoc>
//				( "minDiffRotErr", "threshold for rotation error (radian)", "0.001", "0.", "6.2831854", &P::Comp<T> )
//				( "minDiffTransErr", "threshold for translation error", "0.001", "0.", "inf", &P::Comp<T> )
//				( "smoothLength", "number of iterations over which to average the differencial error", "3", "0", "2147483647", &P::Comp<unsigned> )
//			;
    params["minDiffRotErr"] = "0.00005";
    params["minDiffTransErr"] = "0.0005";
    PM::TransformationChecker* differentialTransformationChecker = PM::get().TransformationCheckerRegistrar.create("DifferentialTransformationChecker", params);
    icp.transformationCheckers.push_back(differentialTransformationChecker);
    params.clear();

    PM::Inspector* nullInspector = PM::get().InspectorRegistrar.create("NullInspector");
//    				( "baseFileName", "base file name for the statistics files (if empty, disabled)", "" )
//				( "dumpPerfOnExit", "dump performance statistics to stderr on exit", "0" )
//				( "dumpStats", "dump the statistics on first and last step", "0" )
    params["baseFileName"] = "performace_stat";
    params["dumpPerfOnExit"] = "1";
    params["dumpStats"] = "1";
//    PM::Inspector* performaceInspector = PM::get().InspectorRegistrar.create("PerformanceInspector", params);
    icp.inspector.reset(nullInspector);
//    icp.inspector.reset(performaceInspector);
    params.clear();
}

void visualize_pcl_point_clouds(PointCloudXYZ::Ptr cloud_slc_template,
                                PointCloudXYZ::Ptr cloud_scene_aligned,
                                PointCloudXYZ::Ptr cloud_scene_not_aligned,
                                std::string extra_info) {
// Visualization
    pcl::visualization::PCLVisualizer viewer("ICP demo");
// Create two verticaly separated viewports
    int v1(0);
    int v2(1);
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

// The color we will be using
    float bckgr_gray_level = 1.0;  // Black
    float txt_gray_lvl = 1.f - bckgr_gray_level;

// Original point cloud is white
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZT> cloud_white_color_h(cloud_slc_template,
                                                                                 (int) 255 * txt_gray_lvl,
                                                                                 (int) 255 * txt_gray_lvl,
                                                                                 (int) 255 * txt_gray_lvl);
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZT> cloud_blue_color_h(cloud_slc_template,
                                                                                 (int) 20,
                                                                                 (int) 20,
                                                                                 (int) 180);
//    viewer.addPointCloud(cloud_slc_template, cloud_white_color_h, "cloud_in_v1", v1);
    viewer.addPointCloud(cloud_slc_template, cloud_blue_color_h, "cloud_in_v1", v1);
//    viewer.addPointCloud(cloud_slc_template, cloud_white_color_h, "cloud_in_v2", v2);
    viewer.addPointCloud(cloud_slc_template, cloud_blue_color_h, "cloud_in_v2", v2);

// Transformed point cloud is red
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZT> cloud_red_color_h(cloud_scene_not_aligned, 180, 20, 20);
    viewer.addPointCloud(cloud_scene_not_aligned, cloud_red_color_h, "cloud_not_aligned_v1", v1);

// ICP aligned point cloud is green
    pcl::visualization::PointCloudColorHandlerCustom<PointXYZT> cloud_green_color_h(cloud_scene_aligned, 20, 180, 20);
    viewer.addPointCloud(cloud_scene_aligned, cloud_green_color_h, "cloud_aligned_v2", v2);

// Adding text descriptions in each viewport
//    viewer.addText("White: template point cloud\nRed: scene not aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_1", v1);
//    viewer.addText("White: template point cloud\nGreen: scene aligned point cloud", 10, 15, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "icp_info_2", v2);

//    std::stringstream ss;
//    ss << n_iteration;
//    std::string iterations_cnt = "ICP n_iteration = " + ss.str();
//    viewer.addText(extra_info, 10, 60, 16, txt_gray_lvl, txt_gray_lvl, txt_gray_lvl, "iterations_cnt", v2);

// Set background color
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

// Set camera position and orientation
    viewer.setCameraPosition(-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(1280, 1024);  // Visualiser window size

// Register keyboard callback :
    viewer.registerKeyboardCallback(&keyboardEventOccurred, (void *) NULL);

// Display the visualiser
    pcl::console::TicToc time;
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
        time.tic();
    }
    viewer.close();

}

void convert_py_array_to_pcl_cloud(__IN int n_point, __IN PyArrayObject * py_array_ptr_cloud, __OUT PointCloudXYZ::Ptr pcl_cloud) {
    for (int i = 0; i < n_point; i++) {
        float * float_ptr_x = (float*) PyArray_GETPTR2(py_array_ptr_cloud, i, 0);
        float * float_ptr_y = (float*) PyArray_GETPTR2(py_array_ptr_cloud, i, 1);
        float * float_prt_z = (float*) PyArray_GETPTR2(py_array_ptr_cloud, i, 2);
        if (isnanf(*float_ptr_x) || isnanf(*float_ptr_y) || isnanf(*float_prt_z))
            continue;
        PointXYZT pt = PointXYZT(*float_ptr_x, *float_ptr_y, *float_prt_z);
        pcl_cloud->push_back(pt);
    }
}

void convert_py_array_xyz_and_rgb_to_pcl_cloud(__IN int n_point,
                                               __IN PyArrayObject * py_array_ptr_cloud_xyz,
                                               __IN PyArrayObject * py_array_ptr_cloud_rgb,
                                               __OUT PointCloudXYZRGB::Ptr pcl_cloud_xyzrgb) {
    for (int i = 0; i < n_point; i++) {
        float * float_ptr_x = (float*) PyArray_GETPTR2(py_array_ptr_cloud_xyz, i, 0);
        float * float_ptr_y = (float*) PyArray_GETPTR2(py_array_ptr_cloud_xyz, i, 1);
        float * float_prt_z = (float*) PyArray_GETPTR2(py_array_ptr_cloud_xyz, i, 2);
        unsigned int * uint32_ptr_rgb = (unsigned int *) PyArray_GETPTR1(py_array_ptr_cloud_rgb, i);
        if (isnanf(*float_ptr_x) || isnanf(*float_ptr_y) || isnanf(*float_prt_z))
            continue;
        pcl::PointXYZRGB pt;
        pt.x = *float_ptr_x;
        pt.y = *float_ptr_y;
        pt.z = *float_prt_z;
        pt.rgb = *reinterpret_cast<float*>(uint32_ptr_rgb);
        pcl_cloud_xyzrgb->push_back(pt);
    }
}

static PyObject * downsamplePointCloudXYZRGB(PyObject * self, PyObject * args) {
    PyObject * py_obj_ptr_cloud_xyz = NULL;
    PyObject * py_obj_ptr_cloud_rgb = NULL;
    PyArrayObject * py_array_ptr_cloud_xyz=NULL;
    PyArrayObject * py_array_ptr_cloud_rgb=NULL;
    float voxel_grid_size;
    if (!PyArg_ParseTuple(args,
                          "OOf",
                          &py_obj_ptr_cloud_xyz,
                          &py_obj_ptr_cloud_rgb,
                          &voxel_grid_size))
        return NULL;
    py_array_ptr_cloud_xyz = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_cloud_xyz, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    py_array_ptr_cloud_rgb = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_cloud_rgb, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_cloud_xyz == NULL) return NULL;
    if (py_array_ptr_cloud_rgb == NULL) return NULL;

    PointCloudXYZRGB::Ptr pcl_cloud_xyzrgb(new PointCloudXYZRGB);  // ICP output point cloudb
    cout << "Setting up pointclouds..." << endl;
    pcl::console::TicToc time;
    time.tic();
    printf("shape of the scene array: \n");
    npy_intp * array_dim_ptr_cloud = PyArray_DIMS(py_array_ptr_cloud_xyz);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_cloud_xyz); i++) {
        printf ("%ld ", array_dim_ptr_cloud[i]);
    }
    printf("\n");
    int n_point_in_cloud_in_beginning = array_dim_ptr_cloud[0];
    convert_py_array_xyz_and_rgb_to_pcl_cloud(n_point_in_cloud_in_beginning, py_array_ptr_cloud_xyz, py_array_ptr_cloud_rgb, pcl_cloud_xyzrgb);
    int n_point_in_cloud_xyzrgb_converted = pcl_cloud_xyzrgb->size();
    printf("N point converted from npy array to pcl cloud: %d\n", n_point_in_cloud_xyzrgb_converted);

    pcl::VoxelGrid<pcl::PointXYZRGB> vox_grid;
    vox_grid.setInputCloud(pcl_cloud_xyzrgb);
    vox_grid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
    //vox_grid.filter (*scene_cloud); // Please see this http://www.pcl-developers.org/Possible-problem-in-new-VoxelGrid-implementation-from-PCL-1-5-0-td5490361.html
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_xyzrbg_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>);
    vox_grid.filter(*pcl_cloud_xyzrbg_downsampled);
    int n_point_in_cloud_xyzrgb_downsampled = pcl_cloud_xyzrbg_downsampled->size();
    printf("N point in down sampled cloud: %d\n", n_point_in_cloud_xyzrgb_downsampled);

    int dims[2];
    dims[0] = n_point_in_cloud_xyzrgb_downsampled;
    dims[1] = 3;
    PyArrayObject * npy_array_xyz_out = (PyArrayObject *) PyArray_FromDims(2,dims, NPY_FLOAT32);
    int point_xyz_stride = PyArray_STRIDES(npy_array_xyz_out)[0];
    PyArrayObject * npy_array_rgb_out = (PyArrayObject *) PyArray_FromDims(1,dims, NPY_UINT32);
//    printf("before assignment");
    int point_rgb_stride = PyArray_STRIDES(npy_array_rgb_out)[0];
    float * dataOfxyz_out = (float *) PyArray_DATA(npy_array_xyz_out);
    float * dataOfrgb_out = (float *) PyArray_DATA(npy_array_rgb_out); // fake float32 array
    for (int i = 0; i < n_point_in_cloud_xyzrgb_downsampled; i++) {
        dataOfxyz_out[i * point_xyz_stride/sizeof(float) + 0] = pcl_cloud_xyzrbg_downsampled->points[i].x;
        dataOfxyz_out[i * point_xyz_stride/sizeof(float) + 1] = pcl_cloud_xyzrbg_downsampled->points[i].y;
        dataOfxyz_out[i * point_xyz_stride/sizeof(float) + 2] = pcl_cloud_xyzrbg_downsampled->points[i].z;
        dataOfrgb_out[i * point_rgb_stride/sizeof(float) + 0] = pcl_cloud_xyzrbg_downsampled->points[i].rgb;
    }
//    printf("after assignment\n");
    PyObject * result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, PyArray_Return(npy_array_xyz_out));
    PyTuple_SetItem(result, 1, PyArray_Return(npy_array_rgb_out));
    Py_DecRef(PyArray_Return(py_array_ptr_cloud_xyz));
    Py_DecRef(PyArray_Return(py_array_ptr_cloud_rgb));
    return result;
//    Py_RETURN_NONE;

////    Py_DecRef(PyArray_Return(py_array_ptr_cloud_xyz));
////    PyArray_XDECREF(py_array_ptr_cloud_xyz);
////    Py_DecRef(PyArray_Return(py_array_ptr_cloud_rgb));
////    PyArray_XDECREF(py_array_ptr_cloud_rgb);
////    PyArray_free(py_array_ptr_cloud_rgb);
//
////    PyArray_free(py_array_ptr_cloud_xyz);
//    printf("ref num: %d\n", PyArray_REFCOUNT(py_array_ptr_cloud_xyz));
//    printf("ref num: %d\n", PyArray_REFCOUNT(py_array_ptr_cloud_rgb));
//    int dims_debug[2];
//    dims_debug[0] = 10;
//    dims_debug[1] = 3;
//    PyArrayObject * npy_array_xyz_out_debug = (PyArrayObject *) PyArray_FromDims(2,dims_debug, NPY_FLOAT32);
//    PyArrayObject * npy_array_rgb_out_debug = (PyArrayObject *) PyArray_FromDims(1,dims_debug, NPY_UINT32);
//    PyObject * result_debug = PyTuple_New(2);
//    PyTuple_SetItem(result_debug, 0, PyArray_Return(npy_array_xyz_out_debug));
//    PyTuple_SetItem(result_debug, 1, PyArray_Return(npy_array_rgb_out_debug));
//    return result_debug;
}

static PyObject * displayTwoPointCloud(PyObject* self, PyObject*args) {
    PyObject * py_obj_ptr_scene_cloud = NULL;
    PyObject * py_obj_ptr_init_translation=NULL;
    PyObject * py_obj_ptr_template_cloud = NULL;
    PyArrayObject * py_array_ptr_scene_cloud=NULL;
    PyArrayObject * py_array_ptr_init_translation=NULL;
    PyArrayObject * py_array_ptr_template_cloud = NULL;
    const char* slc_template_ply_filename_str_ptr;
    const char* scene_pcd_filename_str_prt;
    float template_voxel_dim, scene_voxel_dim;
    int isForce2D, isPointToPlane, isDebug;
    if (!PyArg_ParseTuple(args,
                          "ssOOOffiii",
                          &slc_template_ply_filename_str_ptr,
                          &scene_pcd_filename_str_prt,
                          &py_obj_ptr_scene_cloud,
                          &py_obj_ptr_init_translation,
                          &py_obj_ptr_template_cloud,
                          &template_voxel_dim,
                          &scene_voxel_dim,
                          &isForce2D,
                          &isPointToPlane,
                          &isDebug))
        return NULL;
    py_array_ptr_scene_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_scene_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_scene_cloud == NULL) return NULL;
    py_array_ptr_init_translation = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_init_translation, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_init_translation == NULL) return NULL;
    py_array_ptr_template_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_template_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_template_cloud == NULL) return NULL;
    std::string slc_template_ply_filename(slc_template_ply_filename_str_ptr);
    std::string scene_pcd_filename(scene_pcd_filename_str_prt);

    PointCloudXYZ::Ptr pcl_cloud_slc_template(new PointCloudXYZ);  // Original point cloud
    PointCloudXYZ::Ptr pcl_cloud_scene_aligned(new PointCloudXYZ);  // Transformed point cloud
    PointCloudXYZ::Ptr pcl_cloud_scene_not_aligned(new PointCloudXYZ);  // ICP output point cloud

    cout << "Setting up pointclouds..." << endl;
    pcl::console::TicToc time;
    time.tic();

    // init data for tamplate and scene
    printf("shape of the template array: \n");
    npy_intp * array_dim_ptr_template_cloud = PyArray_DIMS(py_array_ptr_template_cloud);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_template_cloud); i++) {
        printf ("%ld ", array_dim_ptr_template_cloud[i]);
    }
    printf("\n");
    int n_point_in_template_cloud = array_dim_ptr_template_cloud[0];
    convert_py_array_to_pcl_cloud(n_point_in_template_cloud, py_array_ptr_template_cloud, pcl_cloud_slc_template);
    printf("shape of the scene array: \n");
    npy_intp * array_dim_ptr_scene_cloud = PyArray_DIMS(py_array_ptr_scene_cloud);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_scene_cloud); i++) {
        printf ("%ld ", array_dim_ptr_scene_cloud[i]);
    }
    printf("\n");
    int n_point_in_scene_cloud = array_dim_ptr_scene_cloud[0];
    convert_py_array_to_pcl_cloud(n_point_in_scene_cloud, py_array_ptr_scene_cloud, pcl_cloud_scene_not_aligned);

    cout << "Aligning point cloud..." << endl;
    pcl_cloud_slc_template = pipeline_cloud_down_sampling(pcl_cloud_slc_template, template_voxel_dim);
    Eigen::Vector4f template_centroid;
    pcl::compute3DCentroid(*pcl_cloud_slc_template, template_centroid);

    pcl_cloud_scene_not_aligned = pipeline_cloud_down_sampling(pcl_cloud_scene_not_aligned, scene_voxel_dim);
    if (n_point_in_scene_cloud > 1000) {
        pcl_cloud_scene_not_aligned = pipeline_cloud_remove_out_liers(pcl_cloud_scene_not_aligned);
    }
    Eigen::Vector4f scene_centroid;
    pcl::compute3DCentroid(*pcl_cloud_scene_not_aligned, scene_centroid);

    Eigen::Vector4f scene_to_template_translation = scene_centroid - template_centroid;
    scene_to_template_translation(3) = 1.f;
    Eigen::Matrix4f init_scene_transformation = Eigen::Matrix4f::Identity();
    init_scene_transformation.col(3) = scene_to_template_translation;
//    pcl::transformPointCloud(*pcl_cloud_scene_not_aligned, *pcl_cloud_scene_not_aligned, Eigen::Matrix4f(init_scene_transformation.inverse()));

    boost::shared_ptr<DP> pm_ref_ptr = pcl_cloud_to_pm_DataPoints(pcl_cloud_slc_template);
    boost::shared_ptr<DP> pm_data_ptr = pcl_cloud_to_pm_DataPoints(pcl_cloud_scene_not_aligned);
    const DP & ref = *pm_ref_ptr;
    const DP & data = *pm_data_ptr;

    // Create the default ICP algorithm
    PM::ICP icp;

    // See the implementation of setDefault() to create a custom ICP algorithm
//    icp.setDefault();
//    config_icp(icp, isForce2D, isPointToPlane);
    config_icp_general(icp, isForce2D, isPointToPlane);


    // Compute the transformation to express data in ref
//    PM::TransformationParameters T = icp(data, ref);
    PM::TransformationParameters T = init_scene_transformation;//icp(data, ref, init_scene_transformation.inverse());
//    icp.inspector->dumpStats(std::cout);
    PM::TransformationParameters Tinverse = T.inverse();
//    Eigen::Matrix4f scene_to_iiwa = Eigen::Matrix4f::Identity();
//    double theta = M_PI_2;  // The angle of rotation in radians
//    scene_to_iiwa(0, 0) = cos(theta);
//    scene_to_iiwa(0, 1) = -sin(theta);
//    scene_to_iiwa(1, 0) = sin(theta);
//    scene_to_iiwa(1, 1) = cos(theta);

    std::cout << "\nalignment in " << time.toc() << " ms\n" << endl;
    cout << "template to data transformation:" << endl << Tinverse << endl;

    if (isDebug == 1) {
        // Transform data to express it in ref
        DP data_out(data);
        DP data_in(data);
        icp.transformations.apply(data_out, T);
        icp.transformations.apply(data_in, init_scene_transformation.inverse());

        // Safe files to see the results
        //    ref.save("test_ref.vtk");
        PointCloudXYZ::Ptr pcl_ref_cloud_ptr = pm_DataPoints_to_pcl_cloud(ref);
        //    data.save("test_data_in.vtk");
        PointCloudXYZ::Ptr pcl_data_in_cloud_ptr = pm_DataPoints_to_pcl_cloud(data_in);
        //    data_out.save("test_data_out.vtk");
        PointCloudXYZ::Ptr pcl_data_out_cloud_ptr = pm_DataPoints_to_pcl_cloud(data_out);
        visualize_pcl_point_clouds(pcl_ref_cloud_ptr, pcl_data_out_cloud_ptr, pcl_data_in_cloud_ptr, "gby");
    } else {
    }
//    Tinverse = scene_to_iiwa * Tinverse;
    int dims[2];
    dims[0] = 5;
    dims[1] = 4;
    PyArrayObject * Tout = (PyArrayObject *) PyArray_FromDims(2,dims, NPY_FLOAT32);
    float * dataOfTout = (float *) PyArray_DATA(Tout);//Tout->data;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            dataOfTout[i * PyArray_STRIDES(Tout)[0]/sizeof(float) + j] = Tinverse(i, j);
        }
    }
    float overLap = icp.errorMinimizer->getWeightedPointUsedRatio();
    PM::ErrorMinimizer::ErrorElements errorElements = icp.errorMinimizer->getErrorElements();
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 0] = overLap;
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 1] = float(errorElements.averagedMatchingDist2);
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 2] = float(errorElements.weightedMatchingDist2);
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 3] = float(errorElements.matches.dists.cols()); // number of matches (this one is normalized one with pure matches whose weights are non-zero inside)

    return PyArray_Return(Tout);
//    Py_RETURN_NONE;
}

static PyObject * alignTemplateWithSceneICPWithoutMassCentering(PyObject* self, PyObject*args) {
    PyObject * py_obj_ptr_scene_cloud = NULL;
    PyObject * py_obj_ptr_init_translation=NULL;
    PyObject * py_obj_ptr_template_cloud = NULL;
    PyArrayObject * py_array_ptr_scene_cloud=NULL;
    PyArrayObject * py_array_ptr_init_translation=NULL;
    PyArrayObject * py_array_ptr_template_cloud = NULL;
    const char* slc_template_ply_filename_str_ptr;
    const char* scene_pcd_filename_str_prt;
    float template_voxel_dim, scene_voxel_dim;
    int isForce2D, isPointToPlane, isDebug;
    if (!PyArg_ParseTuple(args,
                          "ssOOOffiii",
                          &slc_template_ply_filename_str_ptr,
                          &scene_pcd_filename_str_prt,
                          &py_obj_ptr_scene_cloud,
                          &py_obj_ptr_init_translation,
                          &py_obj_ptr_template_cloud,
                          &template_voxel_dim,
                          &scene_voxel_dim,
                          &isForce2D,
                          &isPointToPlane,
                          &isDebug))
        return NULL;
    py_array_ptr_scene_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_scene_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_scene_cloud == NULL) return NULL;
    py_array_ptr_init_translation = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_init_translation, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_init_translation == NULL) return NULL;
    py_array_ptr_template_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_template_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_template_cloud == NULL) return NULL;

    Py_DecRef(PyArray_Return(py_array_ptr_template_cloud));
    Py_DecRef(PyArray_Return(py_array_ptr_init_translation));
    Py_DecRef(PyArray_Return(py_array_ptr_scene_cloud));

    std::string slc_template_ply_filename(slc_template_ply_filename_str_ptr);
    std::string scene_pcd_filename(scene_pcd_filename_str_prt);

    PointCloudXYZ::Ptr pcl_cloud_slc_template(new PointCloudXYZ);  // Original point cloud
    PointCloudXYZ::Ptr pcl_cloud_scene_aligned(new PointCloudXYZ);  // Transformed point cloud
    PointCloudXYZ::Ptr pcl_cloud_scene_not_aligned(new PointCloudXYZ);  // ICP output point cloud

    cout << "Setting up pointclouds..." << endl;
    pcl::console::TicToc time;
    time.tic();

    // init data for tamplate and scene
    printf("shape of the template array: \n");
    npy_intp * array_dim_ptr_template_cloud = PyArray_DIMS(py_array_ptr_template_cloud);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_template_cloud); i++) {
        printf ("%ld ", array_dim_ptr_template_cloud[i]);
    }
    printf("\n");
    int n_point_in_template_cloud = array_dim_ptr_template_cloud[0];
    convert_py_array_to_pcl_cloud(n_point_in_template_cloud, py_array_ptr_template_cloud, pcl_cloud_slc_template); // _in, _in, _out
    printf("Template non nan point: %d\n", pcl_cloud_slc_template->size());

    printf("shape of the scene array: \n");
    npy_intp * array_dim_ptr_scene_cloud = PyArray_DIMS(py_array_ptr_scene_cloud);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_scene_cloud); i++) {
        printf ("%ld ", array_dim_ptr_scene_cloud[i]);
    }
    printf("\n");
    int n_point_in_scene_cloud = array_dim_ptr_scene_cloud[0];
    convert_py_array_to_pcl_cloud(n_point_in_scene_cloud, py_array_ptr_scene_cloud, pcl_cloud_scene_not_aligned);
    printf("Scene non nan point: %d\n", pcl_cloud_scene_not_aligned->size());

    cout << "Aligning point cloud..." << endl;
    Eigen::Matrix4f init_scene_transformation = Eigen::Matrix4f::Identity();

    boost::shared_ptr<DP> pm_ref_ptr = pcl_cloud_to_pm_DataPoints(pcl_cloud_slc_template);
    boost::shared_ptr<DP> pm_data_ptr = pcl_cloud_to_pm_DataPoints(pcl_cloud_scene_not_aligned);
    const DP & ref = *pm_ref_ptr;
    const DP & data = *pm_data_ptr;

    // Create the default ICP algorithm
    PM::ICP icp;
    config_icp_general(icp, isForce2D, isPointToPlane);

    // Compute the transformation to express data in ref
//    PM::TransformationParameters T = icp(data, ref);
    PM::TransformationParameters T = icp(data, ref, init_scene_transformation.inverse());
//    PM::TransformationParameters T = init_scene_transformation;
//    icp.inspector->dumpStats(std::cout);
    PM::TransformationParameters Tinverse = T.inverse();
    std::cout << "\nalignment in " << time.toc() << " ms\n" << endl;
    cout << "template to data transformation:" << endl << Tinverse << endl;

    if (isDebug == 1) {
        // Transform data to express it in ref
        DP data_out(data);
        DP data_in(data);
        icp.transformations.apply(data_out, T);
        icp.transformations.apply(data_in, init_scene_transformation.inverse());

        // Safe files to see the results
        //    ref.save("test_ref.vtk");
        PointCloudXYZ::Ptr pcl_ref_cloud_ptr = pm_DataPoints_to_pcl_cloud(ref);
        //    data.save("test_data_in.vtk");
        PointCloudXYZ::Ptr pcl_data_in_cloud_ptr = pm_DataPoints_to_pcl_cloud(data_in);
        //    data_out.save("test_data_out.vtk");
        PointCloudXYZ::Ptr pcl_data_out_cloud_ptr = pm_DataPoints_to_pcl_cloud(data_out);
        visualize_pcl_point_clouds(pcl_ref_cloud_ptr, pcl_data_out_cloud_ptr, pcl_data_in_cloud_ptr, "gby");
    } else {
    }
//    Tinverse = scene_to_iiwa * Tinverse;
    int dims[2];
    dims[0] = 5;
    dims[1] = 4;
    PyArrayObject * Tout = (PyArrayObject *) PyArray_FromDims(2,dims, NPY_FLOAT32);
    float * dataOfTout = (float *) PyArray_DATA(Tout);//Tout->data;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            dataOfTout[i * PyArray_STRIDES(Tout)[0]/sizeof(float) + j] = Tinverse(i, j);
        }
    }
    float overLap = icp.errorMinimizer->getWeightedPointUsedRatio();
    PM::ErrorMinimizer::ErrorElements errorElements = icp.errorMinimizer->getErrorElements();
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 0] = overLap;
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 1] = float(errorElements.averagedMatchingDist2);
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 2] = float(errorElements.weightedMatchingDist2);
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 3] = float(errorElements.matches.dists.cols()); // number of matches (this one is normalized one with pure matches whose weights are non-zero inside)

    return PyArray_Return(Tout);
//    Py_RETURN_NONE;
}
static PyObject * alignTemplateWithSceneICP(PyObject* self, PyObject*args) {
    PyObject * py_obj_ptr_scene_cloud = NULL;
    PyObject * py_obj_ptr_init_translation=NULL;
    PyObject * py_obj_ptr_template_cloud = NULL;
    PyArrayObject * py_array_ptr_scene_cloud=NULL;
    PyArrayObject * py_array_ptr_init_translation=NULL;
    PyArrayObject * py_array_ptr_template_cloud = NULL;
    const char* slc_template_ply_filename_str_ptr;
    const char* scene_pcd_filename_str_prt;
    float template_voxel_dim, scene_voxel_dim;
    int isForce2D, isPointToPlane, isDebug;
    if (!PyArg_ParseTuple(args,
                          "ssOOOffiii",
                          &slc_template_ply_filename_str_ptr,
                          &scene_pcd_filename_str_prt,
                          &py_obj_ptr_scene_cloud,
                          &py_obj_ptr_init_translation,
                          &py_obj_ptr_template_cloud,
                          &template_voxel_dim,
                          &scene_voxel_dim,
                          &isForce2D,
                          &isPointToPlane,
                          &isDebug))
        return NULL;
    py_array_ptr_scene_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_scene_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_scene_cloud == NULL) return NULL;
    py_array_ptr_init_translation = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_init_translation, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_init_translation == NULL) return NULL;
    py_array_ptr_template_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_template_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_template_cloud == NULL) return NULL;
    std::string slc_template_ply_filename(slc_template_ply_filename_str_ptr);
    std::string scene_pcd_filename(scene_pcd_filename_str_prt);

    PointCloudXYZ::Ptr pcl_cloud_slc_template(new PointCloudXYZ);  // Original point cloud
    PointCloudXYZ::Ptr pcl_cloud_scene_aligned(new PointCloudXYZ);  // Transformed point cloud
    PointCloudXYZ::Ptr pcl_cloud_scene_not_aligned(new PointCloudXYZ);  // ICP output point cloud

    cout << "Setting up pointclouds..." << endl;
    pcl::console::TicToc time;
    time.tic();

    // init data for tamplate and scene
    printf("shape of the template array: \n");
    npy_intp * array_dim_ptr_template_cloud = PyArray_DIMS(py_array_ptr_template_cloud);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_template_cloud); i++) {
        printf ("%ld ", array_dim_ptr_template_cloud[i]);
    }
    printf("\n");
    int n_point_in_template_cloud = array_dim_ptr_template_cloud[0];
    convert_py_array_to_pcl_cloud(n_point_in_template_cloud, py_array_ptr_template_cloud, pcl_cloud_slc_template); // _in, _in, _out
    printf("Template non nan point: %d\n", pcl_cloud_slc_template->size());

    printf("shape of the scene array: \n");
    npy_intp * array_dim_ptr_scene_cloud = PyArray_DIMS(py_array_ptr_scene_cloud);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_scene_cloud); i++) {
        printf ("%ld ", array_dim_ptr_scene_cloud[i]);
    }
    printf("\n");
    int n_point_in_scene_cloud = array_dim_ptr_scene_cloud[0];
    convert_py_array_to_pcl_cloud(n_point_in_scene_cloud, py_array_ptr_scene_cloud, pcl_cloud_scene_not_aligned);
    printf("Scene non nan point: %d\n", pcl_cloud_scene_not_aligned->size());

    cout << "Aligning point cloud..." << endl;
    pcl_cloud_slc_template = pipeline_cloud_down_sampling(pcl_cloud_slc_template, template_voxel_dim);
    Eigen::Vector4f template_centroid;
    pcl::compute3DCentroid(*pcl_cloud_slc_template, template_centroid);

    pcl_cloud_scene_not_aligned = pipeline_cloud_down_sampling(pcl_cloud_scene_not_aligned, scene_voxel_dim);
    if (n_point_in_scene_cloud > 1000) {
        pcl_cloud_scene_not_aligned = pipeline_cloud_remove_out_liers(pcl_cloud_scene_not_aligned);
    }
    Eigen::Vector4f scene_centroid;
    pcl::compute3DCentroid(*pcl_cloud_scene_not_aligned, scene_centroid);

    Eigen::Vector4f scene_to_template_translation = scene_centroid - template_centroid;
    scene_to_template_translation(3) = 1.f;
    Eigen::Matrix4f init_scene_transformation = Eigen::Matrix4f::Identity();
    init_scene_transformation.col(3) = scene_to_template_translation;
//    pcl::transformPointCloud(*pcl_cloud_scene_not_aligned, *pcl_cloud_scene_not_aligned, Eigen::Matrix4f(init_scene_transformation.inverse()));

    boost::shared_ptr<DP> pm_ref_ptr = pcl_cloud_to_pm_DataPoints(pcl_cloud_slc_template);
    boost::shared_ptr<DP> pm_data_ptr = pcl_cloud_to_pm_DataPoints(pcl_cloud_scene_not_aligned);
    const DP & ref = *pm_ref_ptr;
    const DP & data = *pm_data_ptr;

    // Create the default ICP algorithm
    PM::ICP icp;

    // See the implementation of setDefault() to create a custom ICP algorithm
//    icp.setDefault();
//    config_icp(icp, isForce2D, isPointToPlane);
    config_icp_general(icp, isForce2D, isPointToPlane);


    // Compute the transformation to express data in ref
//    PM::TransformationParameters T = icp(data, ref);
    PM::TransformationParameters T = icp(data, ref, init_scene_transformation.inverse());
//    icp.inspector->dumpStats(std::cout);
    PM::TransformationParameters Tinverse = T.inverse();
//    Eigen::Matrix4f scene_to_iiwa = Eigen::Matrix4f::Identity();
//    double theta = M_PI_2;  // The angle of rotation in radians
//    scene_to_iiwa(0, 0) = cos(theta);
//    scene_to_iiwa(0, 1) = -sin(theta);
//    scene_to_iiwa(1, 0) = sin(theta);
//    scene_to_iiwa(1, 1) = cos(theta);

    std::cout << "\nalignment in " << time.toc() << " ms\n" << endl;
    cout << "template to data transformation:" << endl << Tinverse << endl;

    if (isDebug == 1) {
        // Transform data to express it in ref
        DP data_out(data);
        DP data_in(data);
        icp.transformations.apply(data_out, T);
        icp.transformations.apply(data_in, init_scene_transformation.inverse());

        // Safe files to see the results
    //    ref.save("test_ref.vtk");
        PointCloudXYZ::Ptr pcl_ref_cloud_ptr = pm_DataPoints_to_pcl_cloud(ref);
    //    data.save("test_data_in.vtk");
        PointCloudXYZ::Ptr pcl_data_in_cloud_ptr = pm_DataPoints_to_pcl_cloud(data_in);
    //    data_out.save("test_data_out.vtk");
        PointCloudXYZ::Ptr pcl_data_out_cloud_ptr = pm_DataPoints_to_pcl_cloud(data_out);
        visualize_pcl_point_clouds(pcl_ref_cloud_ptr, pcl_data_out_cloud_ptr, pcl_data_in_cloud_ptr, "gby");
    } else {
    }
//    Tinverse = scene_to_iiwa * Tinverse;
    int dims[2];
    dims[0] = 5;
    dims[1] = 4;
    PyArrayObject * Tout = (PyArrayObject *) PyArray_FromDims(2,dims, NPY_FLOAT32);
    float * dataOfTout = (float *) PyArray_DATA(Tout);//Tout->data;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            dataOfTout[i * PyArray_STRIDES(Tout)[0]/sizeof(float) + j] = Tinverse(i, j);
        }
    }
    float overLap = icp.errorMinimizer->getWeightedPointUsedRatio();
    PM::ErrorMinimizer::ErrorElements errorElements = icp.errorMinimizer->getErrorElements();
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 0] = overLap;
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 1] = float(errorElements.averagedMatchingDist2);
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 2] = float(errorElements.weightedMatchingDist2);
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 3] = float(errorElements.matches.dists.cols()); // number of matches (this one is normalized one with pure matches whose weights are non-zero inside)

    return PyArray_Return(Tout);
//    Py_RETURN_NONE;
}

static PyObject * alignTemplateWithSceneICP2DData(PyObject* self, PyObject*args) {
    PyObject * py_obj_ptr_scene_cloud = NULL, * py_obj_ptr_init_translation=NULL;
    PyObject * py_obj_ptr_template_cloud = NULL;
    PyArrayObject *py_array_ptr_scene_cloud=NULL, *py_array_ptr_init_translation=NULL;
    PyArrayObject *py_array_ptr_template_cloud = NULL;
    const char* slc_template_ply_filename_str_ptr;
    const char* scene_pcd_filename_str_prt;
    float template_voxel_dim, scene_voxel_dim;
    int isForce2D, isPointToPlane, isDebug;
    if (!PyArg_ParseTuple(args, "ssOOOffiii", &slc_template_ply_filename_str_ptr, &scene_pcd_filename_str_prt, &py_obj_ptr_scene_cloud, &py_obj_ptr_init_translation, &py_obj_ptr_template_cloud, &template_voxel_dim, &scene_voxel_dim, &isForce2D, &isPointToPlane, &isDebug))
        return NULL;
    py_array_ptr_scene_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_scene_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_scene_cloud == NULL) return NULL;
    py_array_ptr_init_translation = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_init_translation, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_init_translation == NULL) return NULL;
    py_array_ptr_template_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_template_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_template_cloud == NULL) return NULL;
    std::string slc_template_ply_filename(slc_template_ply_filename_str_ptr);
    std::string scene_pcd_filename(scene_pcd_filename_str_prt);

    PointCloudXYZ::Ptr pcl_cloud_slc_template(new PointCloudXYZ);  // Original point cloud
    PointCloudXYZ::Ptr cloud_scene_aligned(new PointCloudXYZ);  // Transformed point cloud
    PointCloudXYZ::Ptr pcl_cloud_scene_not_aligned(new PointCloudXYZ);  // ICP output point cloud

    cout << "Setting up pointclouds..." << endl;
    pcl::console::TicToc time;
    time.tic();

    // init data for tamplate and scene
    printf("shape of the template array: \n");
    npy_intp * array_dim_ptr_template_cloud = PyArray_DIMS(py_array_ptr_template_cloud);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_template_cloud); i++) {
        printf ("%ld ", array_dim_ptr_template_cloud[i]);
    }
    printf("\n");
    int n_point_in_template_cloud = array_dim_ptr_template_cloud[0];
    convert_py_array_to_pcl_cloud(n_point_in_template_cloud, py_array_ptr_template_cloud, pcl_cloud_slc_template);
    printf("shape of the scene array: \n");
    npy_intp * array_dim_ptr_scene_cloud = PyArray_DIMS(py_array_ptr_scene_cloud);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_scene_cloud); i++) {
        printf ("%ld ", array_dim_ptr_scene_cloud[i]);
    }
    printf("\n");
    int n_point_in_scene_cloud = array_dim_ptr_scene_cloud[0];
    convert_py_array_to_pcl_cloud(n_point_in_scene_cloud, py_array_ptr_scene_cloud, pcl_cloud_scene_not_aligned);

    cout << "Aligning point cloud..." << endl;
//    pcl_cloud_slc_template = pipeline_cloud_down_sampling(pcl_cloud_slc_template, 0.001f);
    pcl_cloud_slc_template = pipeline_cloud_down_sampling(pcl_cloud_slc_template, template_voxel_dim);
    Eigen::Vector4f template_centroid;
    pcl::compute3DCentroid(*pcl_cloud_slc_template, template_centroid);

    pcl_cloud_scene_not_aligned = pipeline_cloud_down_sampling(pcl_cloud_scene_not_aligned, scene_voxel_dim);
    if (n_point_in_scene_cloud > 1000) {
        pcl_cloud_scene_not_aligned = pipeline_cloud_remove_out_liers(pcl_cloud_scene_not_aligned);
    }
    Eigen::Vector4f scene_centroid;
    pcl::compute3DCentroid(*pcl_cloud_scene_not_aligned, scene_centroid);

    Eigen::Vector4f scene_to_template_translation = scene_centroid - template_centroid;
    scene_to_template_translation(3) = 1.f;
    Eigen::Matrix4f init_scene_transformation = Eigen::Matrix4f::Identity();
    init_scene_transformation.col(3) = scene_to_template_translation;
//    pcl::transformPointCloud(*pcl_cloud_scene_not_aligned, *pcl_cloud_scene_not_aligned, Eigen::Matrix4f(init_scene_transformation.inverse()));


    Eigen::Matrix3f init_scene_transformation_2d = Eigen::Matrix3f::Identity();
    init_scene_transformation_2d.topLeftCorner(2,2) = init_scene_transformation.topLeftCorner(2,2);
//    init_scene_transformation_2d.col(2) = init_scene_transformation.col(3).head(3)
    init_scene_transformation_2d.col(2).head(2) = init_scene_transformation.col(3).head(2);
    cout << "init_scene_tranaformation_2d:"<<endl;
    cout << init_scene_transformation_2d<<endl;
    boost::shared_ptr<DP> ref_ptr = pcl_cloud_to_pm_DataPoints(pcl_cloud_slc_template);
    boost::shared_ptr<DP> data_ptr = pcl_cloud_to_pm_DataPoints(pcl_cloud_scene_not_aligned);
    const DP & ref = *ref_ptr;
    const DP & data = *data_ptr;
    DP * ref_2d_ptr = pcl_cloud_to_pm_DataPoints_2D_data(pcl_cloud_slc_template);
    DP * data_2d_ptr = pcl_cloud_to_pm_DataPoints_2D_data(pcl_cloud_scene_not_aligned);
    const DP & ref_2d = *ref_2d_ptr;
    const DP & data_2d = *data_2d_ptr;

    // Create the default ICP algorithm
    PM::ICP icp;

    // See the implementation of setDefault() to create a custom ICP algorithm
//    icp.setDefault();
    config_icp(icp, isForce2D, isPointToPlane);

    // Compute the transformation to express data in ref
//    PM::TransformationParameters T = icp(data, ref, init_scene_transformation.inverse());
    PM::TransformationParameters T_2d = icp(data_2d, ref_2d, init_scene_transformation_2d.inverse());
    PM::TransformationParameters T = Eigen::Matrix4f::Identity();
    T.topLeftCorner(2,2) = T_2d.topLeftCorner(2,2);
    T.topRightCorner(2, 1) = T_2d.topRightCorner(2,1);
    T(2, 3) = -init_scene_transformation(2, 3);
    PM::TransformationParameters Tinverse = T.inverse();
    Eigen::Matrix4f scene_to_iiwa = Eigen::Matrix4f::Identity();
//    double theta = M_PI_2;  // The angle of rotation in radians
//    scene_to_iiwa(0, 0) = cos(theta);
//    scene_to_iiwa(0, 1) = -sin(theta);
//    scene_to_iiwa(1, 0) = sin(theta);
//    scene_to_iiwa(1, 1) = cos(theta);

    std::cout << "\nalignment in " << time.toc() << " ms\n" << endl;
    cout << "template to data transformation:" << endl << Tinverse << endl;

    if (isDebug == 1) {
        // Transform data to express it in ref
        DP data_out(data);
        DP data_in(data);
        icp.transformations.apply(data_out, T);
        icp.transformations.apply(data_in, init_scene_transformation.inverse());

        // Safe files to see the results
    //    ref.save("test_ref.vtk");
        PointCloudXYZ::Ptr pcl_ref_cloud_ptr = pm_DataPoints_to_pcl_cloud(ref);
    //    data.save("test_data_in.vtk");
        PointCloudXYZ::Ptr pcl_data_in_cloud_ptr = pm_DataPoints_to_pcl_cloud(data_in);
    //    data_out.save("test_data_out.vtk");
        PointCloudXYZ::Ptr pcl_data_out_cloud_ptr = pm_DataPoints_to_pcl_cloud(data_out);
        visualize_pcl_point_clouds(pcl_ref_cloud_ptr, pcl_data_out_cloud_ptr, pcl_data_in_cloud_ptr, "gby");
    } else {
    }
//    Tinverse = scene_to_iiwa * Tinverse;
    int dims[2];
    dims[0] = 5;
    dims[1] = 4;
    PyArrayObject * Tout = (PyArrayObject *) PyArray_FromDims(2,dims, NPY_FLOAT32);
    float * dataOfTout = (float *) PyArray_DATA(Tout);//Tout->data;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            dataOfTout[i * PyArray_STRIDES(Tout)[0]/sizeof(float) + j] = Tinverse(i, j);
        }
    }
    float overLap = icp.errorMinimizer->getWeightedPointUsedRatio();
    PM::ErrorMinimizer::ErrorElements errorElements = icp.errorMinimizer->getErrorElements();
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 0] = overLap;
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 1] = float(errorElements.averagedMatchingDist2);
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 2] = float(errorElements.weightedMatchingDist2);
    dataOfTout[4 * PyArray_STRIDES(Tout)[0]/sizeof(float) + 3] = float(errorElements.matches.dists.cols()); // number of matches (this one is normalized one with pure matches whose weights are non-zero inside)

    return PyArray_Return(Tout);
//    Py_RETURN_NONE;
}

static PyObject* say_hello(PyObject* self, PyObject* args)
{
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name))
        return NULL;
    printf("Hello %s!\n", name);
    Py_RETURN_NONE;
}

static PyObject* print_ndarray_info(PyObject* self, PyObject* args)
{
    PyObject * py_obj_ptr_scene_cloud = NULL;
    PyArrayObject *py_array_ptr_scene_cloud=NULL;
    if (!PyArg_ParseTuple(args, "O", &py_obj_ptr_scene_cloud))
        return NULL;
//    py_array_ptr_scene_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_scene_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    py_array_ptr_scene_cloud = (PyArrayObject *)PyArray_FROM_OTF(py_obj_ptr_scene_cloud, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (py_array_ptr_scene_cloud == NULL) return NULL;
    printf("shape of the array: \n");
    npy_intp * array_dim_ptr = PyArray_DIMS(py_array_ptr_scene_cloud);
    for (int i = 0; i < PyArray_NDIM(py_array_ptr_scene_cloud); i++) {
        printf ("%ld ", array_dim_ptr[i]);
    }
    printf("\n");
    if (PyArray_NDIM(py_array_ptr_scene_cloud) == 1) {
        printf ("1d array values:\n");
        for (int i = 0; i < array_dim_ptr[0]; i++) {
            float * pValue = (float*)PyArray_GETPTR1(py_array_ptr_scene_cloud, i);
            printf ("%d: %f\n", i, *pValue);
        }
        printf("\n");
    }
    Py_RETURN_NONE;
}

static PyMethodDef HelloMethods[] =
{
     {"say_hello", say_hello, METH_VARARGS, "Greet somebody."},
     {"alignTemplateWithSceneICP", alignTemplateWithSceneICP, METH_VARARGS, " align two point clouds"},
     {"alignTemplateWithSceneICPWithoutMassCentering", alignTemplateWithSceneICPWithoutMassCentering, METH_VARARGS, " align two point clouds"},
     {"displayTwoPointCloud", displayTwoPointCloud, METH_VARARGS, " display two point clouds"},
     {"alignTemplateWithSceneICP2DData", alignTemplateWithSceneICP2DData, METH_VARARGS, " align two point clouds in x-y plane"},
     {"downsamplePointCloudXYZRGB", downsamplePointCloudXYZRGB, METH_VARARGS, "Downsample the xyzrgb point cloud"},
     {"print_ndarray_info", print_ndarray_info, METH_VARARGS, "print ndarray info"},
     {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC

initlibpointmatcherPythonWrapper(void)
{
     (void) Py_InitModule("libpointmatcherPythonWrapper", HelloMethods);
     import_array();
//     printf("n threads %d.", omp_get_num_threads());
//     omp_set_num_threads(4);
//     printf ("OpenMp num proc %d.\n", omp_get_num_procs());
//     printf ("OpenMp max num threads %d.\n", omp_get_max_threads());
}