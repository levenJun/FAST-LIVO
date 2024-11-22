#include "lidar_selection.h"

namespace lidar_selection {

LidarSelector::LidarSelector(const int gridsize, SparseMap* sparsemap ): grid_size(gridsize), sparse_map(sparsemap)
{
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    G = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    H_T_H = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    Rli = M3D::Identity();
    Rci = M3D::Identity();
    Rcw = M3D::Identity();
    Jdphi_dR = M3D::Identity();
    Jdp_dt = M3D::Identity();
    Jdp_dR = M3D::Identity();
    Pli = V3D::Zero();
    Pci = V3D::Zero();
    Pcw = V3D::Zero();
    width = 800;
    height = 600;
}

LidarSelector::~LidarSelector() 
{
    delete sparse_map;
    delete sub_sparse_map;
    delete[] grid_num;
    delete[] map_index;
    delete[] map_value;
    unordered_map<int, Warp*>().swap(Warp_map);
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);
    unordered_map<VOXEL_KEY, VOXEL_POINTS*>().swap(feat_map);  
}

void LidarSelector::set_extrinsic(const V3D &transl, const M3D &rot)
{
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

void LidarSelector::init()
{
    sub_sparse_map = new SubSparseMap;
    Rci = sparse_map->Rcl * Rli;
    Pci= sparse_map->Rcl*Pli + sparse_map->Pcl;
    M3D Ric;
    V3D Pic;
    Jdphi_dR = Rci;
    Pic = -Rci.transpose() * Pci;
    M3D tmp;
    tmp << SKEW_SYM_MATRX(Pic);
    Jdp_dR = -Rci * tmp;
    width = cam->width();
    height = cam->height();
    grid_n_width = static_cast<int>(width/grid_size);
    grid_n_height = static_cast<int>(height/grid_size);
    length = grid_n_width * grid_n_height;
    fx = cam->errorMultiplier2();
    fy = cam->errorMultiplier() / (4. * fx);
    grid_num = new int[length];
    map_index = new int[length];
    map_value = new float[length];
    map_dist = (float*)malloc(sizeof(float)*length);
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    memset(map_value, 0, sizeof(float)*length);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
    patch_size_total = patch_size * patch_size;
    patch_size_half = static_cast<int>(patch_size/2);
    patch_cache.resize(patch_size_total);
    stage_ = STAGE_FIRST_FRAME;
    pg_down.reset(new PointCloudXYZI());
    weight_scale_ = 10;
    weight_function_.reset(new vk::robust_cost::HuberWeightFunction());
    // weight_function_.reset(new vk::robust_cost::TukeyWeightFunction());
    scale_estimator_.reset(new vk::robust_cost::UnitScaleEstimator());
    // scale_estimator_.reset(new vk::robust_cost::MADScaleEstimator());
}

void LidarSelector::reset_grid()
{
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    fill_n(map_dist, length, 10000);
    std::vector<PointPtr>(length).swap(voxel_points_);
    std::vector<V3D>(length).swap(add_voxel_points_);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
}

//uv对相机系坐标Pc的雅可比
void LidarSelector::dpi(V3D p, MD(2,3)& J) {
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1./p[2];
    const double z_inv_2 = z_inv * z_inv;
    J(0,0) = fx * z_inv;
    J(0,1) = 0.0;
    J(0,2) = -fx * x * z_inv_2;
    J(1,0) = 0.0;
    J(1,1) = fy * z_inv;
    J(1,2) = -fy * y * z_inv_2;
}

float LidarSelector::CheckGoodPoints(cv::Mat img, V2D uv)
{
    const float u_ref = uv[0];
    const float v_ref = uv[1];
    const int u_ref_i = floorf(uv[0]); 
    const int v_ref_i = floorf(uv[1]);
    const float subpix_u_ref = u_ref-u_ref_i;
    const float subpix_v_ref = v_ref-v_ref_i;
    uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i)*width + (u_ref_i);
    float gu = 2*(img_ptr[1] - img_ptr[-1]) + img_ptr[1-width] - img_ptr[-1-width] + img_ptr[1+width] - img_ptr[-1+width];
    float gv = 2*(img_ptr[width] - img_ptr[-width]) + img_ptr[width+1] - img_ptr[-width+1] + img_ptr[width-1] - img_ptr[-width-1];
    return fabs(gu)+fabs(gv);
}

//以pc为中心, 获取图像img的正方形patch块
void LidarSelector::getpatch(cv::Mat img, V2D pc, float* patch_tmp, int level) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale =  (1<<level);
    const int u_ref_i = floorf(pc[0]/scale)*scale; 
    const int v_ref_i = floorf(pc[1]/scale)*scale;
    const float subpix_u_ref = (u_ref-u_ref_i)/scale;
    const float subpix_v_ref = (v_ref-v_ref_i)/scale;
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    for (int x=0; x<patch_size; x++) 
    {
        uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i-patch_size_half*scale+x*scale)*width + (u_ref_i-patch_size_half*scale);
        for (int y=0; y<patch_size; y++, img_ptr+=scale)
        {
            patch_tmp[patch_size_total*level+x*patch_size+y] = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[scale] + w_ref_bl*img_ptr[scale*width] + w_ref_br*img_ptr[scale*width+scale];
        }
    }
}

// lidar点投影到当前网格,新建 Point点, Feature观测, 并将新建Point点添加进全局体素地图 feat_map
// [单个grid内已经匹配到的体素地图点,如果响应值被超过,也会直接新建point]
// 1)将原始lidar帧所有点全部投影到当前图像网格内,单个网格保留harris值最大的投影点
// 2)有效投影网格都新建 Point点, Feature观测, 并添加到当前 new_frame_中
// 3)将Point点添加进全局体素地图 feat_map
void LidarSelector::addSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg) 
{
    // double t0 = omp_get_wtime();
    reset_grid();

    // double t_b1 = omp_get_wtime() - t0;
    // t0 = omp_get_wtime();
    
    for (int i=0; i<pg->size(); i++) 
    {
        V3D pt(pg->points[i].x, pg->points[i].y, pg->points[i].z);
        V2D pc(new_frame_->w2c(pt));
        if(new_frame_->cam_->isInFrame(pc.cast<int>(), (patch_size_half+1)*8)) // 20px is the patch size in the matcher
        {
            int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
            // float cur_value = CheckGoodPoints(img, pc);
            float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);//harris 角点响应值

            if (cur_value > map_value[index]) //&& (grid_num[index] != TYPE_MAP || map_value[index]<=10)) //! only add in not occupied grid
            {
                map_value[index] = cur_value;
                add_voxel_points_[index] = pt;
                grid_num[index] = TYPE_POINTCLOUD;
            }
        }
    }

    // double t_b2 = omp_get_wtime() - t0;
    // t0 = omp_get_wtime();
    
    int add=0;
    for (int i=0; i<length; i++) 
    {
        if (grid_num[i]==TYPE_POINTCLOUD)// && (map_value[i]>=10)) //! debug
        {
            V3D pt = add_voxel_points_[i];
            V2D pc(new_frame_->w2c(pt));

            PointPtr pt_new(new Point(pt));
            Vector3d f = cam->cam2world(pc);
            FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_, map_value[i], 0));
            ftr_new->img = new_frame_->img_pyr_[0];
            // ftr_new->ImgPyr.resize(5);
            // for(int i=0;i<5;i++) ftr_new->ImgPyr[i] = new_frame_->img_pyr_[i];
            ftr_new->id_ = new_frame_->id_;

            pt_new->addFrameRef(ftr_new);
            pt_new->value = map_value[i];
            AddPoint(pt_new);
            add += 1;
        }
    }

    // double t_b3 = omp_get_wtime() - t0;

    printf("[ VIO ]: Add %d 3D points.\n", add);
    // printf("pg.size: %d \n", pg->size());
    // printf("B1. : %.6lf \n", t_b1);
    // printf("B2. : %.6lf \n", t_b2);
    // printf("B3. : %.6lf \n", t_b3);
}

//将 pt_new 添加进全局体素地图 feat_map
void LidarSelector::AddPoint(PointPtr pt_new)
{
    V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    double voxel_size = 0.5;
    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
      loc_xyz[j] = pt_w[j] / voxel_size;
      if(loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_KEY position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if(iter != feat_map.end())
    {
      iter->second->voxel_points.push_back(pt_new);
      iter->second->count++;
    }
    else
    {
      VOXEL_POINTS *ot = new VOXEL_POINTS(0);
      ot->voxel_points.push_back(pt_new);
      feat_map[position] = ot;
    }
}
// 已知相机模型 cam
// 已知参考 patch块对应的中心像素px_ref,相机投影方向f_ref,中心像素深度depth_ref, 
// 已知当前帧位姿 T_cur_ref
// 计算得到 参考patch到当前帧的仿射A
void LidarSelector::getWarpMatrixAffine(
    const vk::AbstractCamera& cam,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,    // the corresponding pyrimid level of px_ref
    const int pyramid_level,
    const int halfpatch_size,
    Matrix2d& A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref*depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)*(1<<pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)*(1<<pyramid_level)));
//   Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
//   Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
  xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref*(xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref*(xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref*(xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

void LidarSelector::warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int pyramid_level,
    const int halfpatch_size,
    float* patch)
{
  const int patch_size = halfpatch_size*2 ;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if(isnan(A_ref_cur(0,0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }
//   Perform the warp on a larger patch.
//   float* patch_ptr = patch;
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref) / (1<<pyramid_level);
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
  for (int y=0; y<patch_size; ++y)
  {
    for (int x=0; x<patch_size; ++x)//, ++patch_ptr)
    {
      // P[patch_size_total*level + x*patch_size+y]
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
      px_patch *= (1<<search_level);
      px_patch *= (1<<pyramid_level);
      const Vector2f px(A_ref_cur*px_patch + px_ref.cast<float>());
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
        patch[patch_size_total*pyramid_level + y*patch_size+x] = 0;
        // *patch_ptr = 0;
      else
        patch[patch_size_total*pyramid_level + y*patch_size+x] = (float) vk::interpolateMat_8u(img_ref, px[0], px[1]);
        // *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

double LidarSelector::NCC(float* ref_patch, float* cur_patch, int patch_size)
{    
    double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
    double mean_ref =  sum_ref / patch_size;

    double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
    double mean_curr =  sum_cur / patch_size;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patch_size; i++) 
    {
        double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
        numerator += n;
        demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
        demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

//已知仿射阵[塔0层],求最佳塔层
int LidarSelector::getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();

  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

#ifdef FeatureAlign
void LidarSelector::createPatchFromPatchWithBorder(float* patch_with_border, float* patch_ref)
{
  float* ref_patch_ptr = patch_ref;
  for(int y=1; y<patch_size+1; ++y, ref_patch_ptr += patch_size)
  {
    float* ref_patch_border_ptr = patch_with_border + y*(patch_size+2) + 1;
    for(int x=0; x<patch_size; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}
#endif

// img:当前帧图像?
// pg:当前帧lidar点云?
// 当前帧图像划分grid网格后,与体素地图做patch匹配,结果缓存进 sub_sparse_map

// 1,当前lidar帧生成深度图, 并检索到candi体素列表
    //1)将输入的点云 pg 降采样后, 填充被刚才清空的hash体素 sub_feat_map, 作为lidar帧检索到的体素:[只检索lidar点末端体素][未检索经过的体素]
    //2)将输入的点云 pg 降采样后投影到当前帧上,得到深度图depth_img
// 2,当前图像划分网格并通过投影建立和体素点的匹配.单个网格有多个匹配,取深度最小的体素点.
    // 1)当前图像划分网格并建立和体素点的匹配.单个网格有多个匹配,取深度最小的体素点.
// 3,剔除错误匹配 并计算 单个grid到匹配的体素点的patch块的仿射变换,结果缓存到 sub_sparse_map
    //1)遍历匹配对,计算体素点(并取最佳ref patch)到当前帧的仿射变换,计算ref patch逆仿射patch块
    //2)剔除误匹配:a)当前patch在当前深度图上深度不连续;b)当前patch块和ref patch逆仿射块 NCC或者灰度差异过大;c)当前patch观测视角和参考patch观测视角差异过大
void LidarSelector::addFromSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg)
{
    if(feat_map.size()<=0) return;
    // double ts0 = omp_get_wtime();

    pg_down->reserve(feat_map.size());
    downSizeFilter.setInputCloud(pg);
    downSizeFilter.filter(*pg_down);
    
    reset_grid();
    memset(map_value, 0, sizeof(float)*length);

    sub_sparse_map->reset();                        //清空当前单帧到局部地图的匹配?
    deque< PointPtr >().swap(sub_map_cur_frame_);   //清空当前单帧到局部地图的匹配

    float voxel_size = 0.5;
    
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map); //清空 当前lidar帧检索到的体素列表  //清空 hash体素 sub_feat_map
    unordered_map<int, Warp*>().swap(Warp_map);           //清空 匹配到的ref patch块到当前帧的仿射阵缓存

    cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);//当前lidar帧投影到当前图像内的深度图
    float* it = (float*)depth_img.data;

    double t_insert, t_depth, t_position;
    t_insert=t_depth=t_position=0;

    int loc_xyz[3];

    // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
    // double ts1 = omp_get_wtime();

    //1)将输入的点云 pg 降采样后, 填充被刚才清空的hash体素 sub_feat_map, 作为lidar帧检索到的体素:[只检索lidar点末端体素][未检索经过的体素]
    //2)将输入的点云 pg 降采样后投影到当前帧上,得到深度图depth_img
    for(int i=0; i<pg_down->size(); i++)
    {
        // Transform Point to world coordinate
        V3D pt_w(pg_down->points[i].x, pg_down->points[i].y, pg_down->points[i].z);

        // Determine the key of hash table      
        for(int j=0; j<3; j++)
        {
            loc_xyz[j] = floor(pt_w[j] / voxel_size);//体素边长0.5米?
        }
        VOXEL_KEY position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);//将地图点坐标量化后得到hash坐标索引

        auto iter = sub_feat_map.find(position);
        if(iter == sub_feat_map.end())
        {
            sub_feat_map[position] = 1.0;//检索到的体素
        }
                    
        V3D pt_c(new_frame_->w2f(pt_w));//将地图点坐标转到当前帧Pc

        V2D px;
        if(pt_c[2] > 0)
        {
            px[0] = fx * pt_c[0]/pt_c[2] + cx;
            px[1] = fy * pt_c[1]/pt_c[2] + cy;

            if(new_frame_->cam_->isInFrame(px.cast<int>(), (patch_size_half+1)*8))
            {
                float depth = pt_c[2];
                int col = int(px[0]);
                int row = int(px[1]);
                it[width*row+col] = depth;//投影得到的lidar深度
            }
        }
    }
    
    // imshow("depth_img", depth_img);
    // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
    // printf("A11. calculate pt position: %.6lf \n", t_position);
    // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
    // printf("A13. generate depth map: %.6lf \n", t_depth);
    // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);
    

    // double t1 = omp_get_wtime();
    
    // 1)当前图像划分网格并建立和体素点的匹配.单个网格有多个匹配,取深度最小的体素点.

    // lidar帧检索到candi体素 sub_feat_map.
    // 当前图像帧重新划分网格grid, 将检索到的candi体素内的点都投影到grid上.
    // 1)刷新网格的内容标记 grid_num; 2)刷新网格的最小投影深度 map_dist; 3)刷新 网格最小深度投影点对应的 体素点 voxel_points_
    for(auto& iter : sub_feat_map)//遍历lidar检索到的体素
    {   
        VOXEL_KEY position = iter.first;
        // double t4 = omp_get_wtime();
        auto corre_voxel = feat_map.find(position);//当前lidar帧可视范围内的hash体素
        // double t5 = omp_get_wtime();

        if(corre_voxel != feat_map.end())
        {
            std::vector<PointPtr> &voxel_points = corre_voxel->second->voxel_points;
            int voxel_num = voxel_points.size();
            for (int i=0; i<voxel_num; i++)
            {
                PointPtr pt = voxel_points[i];
                if(pt==nullptr) continue;
                V3D pt_cam(new_frame_->w2f(pt->pos_));
                if(pt_cam[2]<0) continue;               //体素内地图点要求在当前帧前方

                V2D pc(new_frame_->w2c(pt->pos_));      //将地图点坐标转到当前帧图像uv

                FeaturePtr ref_ftr;
      
                if(new_frame_->cam_->isInFrame(pc.cast<int>(), (patch_size_half+1)*8)) // 20px is the patch size in the matcher
                {
                    int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
                    grid_num[index] = TYPE_MAP;                     //体素地图点投影到当前网格,当前网格标记为MAP点
                    Vector3d obs_vec(new_frame_->pos() - pt->pos_); //体素地图点观测方向向量

                    float cur_dist = obs_vec.norm();
                    float cur_value = pt->value;

                    if (cur_dist <= map_dist[index]) 
                    {
                        map_dist[index] = cur_dist;//单个网格内取深度最小的体素点
                        voxel_points_[index] = pt;
                    } 

                    if (cur_value >= map_value[index])
                    {
                        map_value[index] = cur_value;
                    }
                }
            }    
        } 
    }
        
    // double t2 = omp_get_wtime();

    // cout<<"B. feat_map.find: "<<t2-t1<<endl;

    double t_2, t_3, t_4, t_5;
    t_2=t_3=t_4=t_5=0;

    //1)遍历匹配对,计算体素点(并取最佳ref patch)到当前帧的仿射变换,计算ref patch逆仿射patch块
    //2)剔除误匹配:a)当前patch在当前深度图上深度不连续;b)当前patch块和ref patch逆仿射块 NCC或者灰度差异过大;c)当前patch观测视角和参考patch观测视角差异过大
    for (int i=0; i<length; i++) //遍历当前图所有网格?
    { 
        if (grid_num[i]==TYPE_MAP) //&& map_value[i]>10)
        {
            // double t_1 = omp_get_wtime();

            PointPtr pt = voxel_points_[i];

            if(pt==nullptr) continue;

            V2D pc(new_frame_->w2c(pt->pos_));                  //体素地图点投影到图像的像素坐标
            V3D pt_cam(new_frame_->w2f(pt->pos_));              //体素地图点投影到当前帧的3d坐标
   
            bool depth_continous = false;//fasle代表深度连续: 自己和patch范围内所有点深度差都在1.5米以下
            for (int u=-patch_size_half; u<=patch_size_half; u++)
            {
                for (int v=-patch_size_half; v<=patch_size_half; v++)
                {
                    if(u==0 && v==0) continue;

                    float depth = it[width*(v+int(pc[1]))+u+int(pc[0])];

                    if(depth == 0.) continue;

                    double delta_dist = abs(pt_cam[2]-depth);

                    if(delta_dist > 1.5)
                    {                
                        depth_continous = true;
                        break;
                    }
                }
                if(depth_continous) break;
            }
            if(depth_continous) continue;//深度不连续

            // t_2 += omp_get_wtime() - t_1;

            // t_1 = omp_get_wtime();
            
            FeaturePtr ref_ftr;

            if(!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;//当前patch和ref patch观测体素点要求观测视角最接近, 搜索结果为 ref_ftr
                                                                              //疑问:如果pt此时观测帧为空,怎么办?

            // t_3 += omp_get_wtime() - t_1;

            std::vector<float> patch_wrap(patch_size_total * 3);//参考patch图像对应的逆向仿射块[3层]

            // patch_wrap = ref_ftr->patch;

            // t_1 = omp_get_wtime();
            // 同一个激光点有唯一世界坐标, 不同相机有不同位姿,观测到不同的patch像素块
            // 以相同激光点为跳板, 不同相机的patch块之间可以得到不同的仿射阵
            // 所以激光点有世界坐标
            // 所以单个patch块, 有相机pose, 有对应的像素中心点, 另外还观测到唯一激光点
            // 所以两个不同patch块间能计算出不同的仿射阵

            int search_level;
            Matrix2d A_cur_ref_zero;

            //[疑问]: 这里仿射阵都应该需要实时计算, 缓存的仿射阵是错的吧?
                // A: 这里是缓存单帧范围内计算的,减少重复计算
            auto iter_warp = Warp_map.find(ref_ftr->id_);
            if(iter_warp != Warp_map.end())//当前帧已经匹配过这个ref patch, 并计算过仿射阵了
            {
                search_level = iter_warp->second->search_level;
                A_cur_ref_zero = iter_warp->second->A_cur_ref;
            }
            else
            {
                // 计算得到 参考patch到当前帧的仿射阵: A_cur_ref_zero
                getWarpMatrixAffine(*cam, ref_ftr->px, ref_ftr->f, (ref_ftr->pos() - pt->pos_).norm(), 
                new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0, patch_size_half, A_cur_ref_zero);
                
                // 已知仿射阵[塔0层],求最佳塔层 [不太懂原理]
                search_level = getBestSearchLevel(A_cur_ref_zero, 2);

                Warp *ot = new Warp(search_level, A_cur_ref_zero);
                Warp_map[ref_ftr->id_] = ot;
            }

            // t_4 += omp_get_wtime() - t_1;

            // t_1 = omp_get_wtime();

            // 已经得到参考ref_ftr patch到当前帧pose的仿射阵
            // 现在得到当前帧到ref块反向仿射变换,并得到参考块图像对应的逆向仿射灰度块
            // 3层金字塔都计算,结果到 patch_wrap
            for(int pyramid_level=0; pyramid_level<=2; pyramid_level++)
            {                
                warpAffine(A_cur_ref_zero, ref_ftr->img, ref_ftr->px, ref_ftr->level, search_level, pyramid_level, patch_size_half, patch_wrap.data());
            }

            //以pc为中心, 获取当前图像img的正方形patch块-->patch_cache [塔0层]
            getpatch(img, pc, patch_cache.data(), 0);

            // 用ncc或者直接暴力对比 参考patch仿射块和当前正方形patch块灰度值
            // 剔除错误匹配
            if(ncc_en)
            {
                double ncc = NCC(patch_wrap.data(), patch_cache.data(), patch_size_total);
                if(ncc < ncc_thre) continue;
            }

            float error = 0.0;
            for (int ind=0; ind<patch_size_total; ind++) 
            {
                error += (patch_wrap[ind]-patch_cache[ind]) * (patch_wrap[ind]-patch_cache[ind]);
            }
            if(error > outlier_threshold*patch_size_total) continue;
            
            sub_map_cur_frame_.push_back(pt);

            sub_sparse_map->propa_errors.push_back(error);
            sub_sparse_map->search_levels.push_back(search_level);
            sub_sparse_map->errors.push_back(error);
            sub_sparse_map->index.push_back(i);                         //当前帧网格idx
            sub_sparse_map->voxel_points.push_back(pt);                 //匹配到的体素地图点
            sub_sparse_map->patch.push_back(std::move(patch_wrap));     //匹配到的体素地图点的参考patch块,逆仿射块[3层]
            // t_5 += omp_get_wtime() - t_1;
        }
    }
    // double t3 = omp_get_wtime();
    // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
    // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4 "<<t_5<<endl;
    printf("[ VIO ]: choose %d points from sub_sparse_map.\n", int(sub_sparse_map->index.size()));
}

#ifdef FeatureAlign
bool LidarSelector::align2D(
    const cv::Mat& cur_img,
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    int index)
{
#ifdef __ARM_NEON__
  if(!no_simd)
    return align2D_NEON(cur_img, ref_patch_with_border, ref_patch, n_iter, cur_px_estimate);
#endif

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;
  const int patch_area_ = 64;
  bool converged=false;

  // compute derivative of template and prepare inverse compositional
  float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
  float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
  Matrix3f H; H.setZero();

  // compute gradient and hessian
  const int ref_step = patch_size_+2;
  float* it_dx = ref_patch_dx;
  float* it_dy = ref_patch_dy;
  for(int y=0; y<patch_size_; ++y) 
  {
    float* it = ref_patch_with_border + (y+1)*ref_step + 1; 
    for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
    {
      Vector3f J;
      J[0] = 0.5 * (it[1] - it[-1]); 
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]); 
      J[2] = 1; 
      *it_dx = J[0];
      *it_dy = J[1];
      H += J*J.transpose(); 
    }
  }
  Matrix3f Hinv = H.inverse();
  float mean_diff = 0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03*0.03;//0.03*0.03
  const int cur_step = cur_img.step.p[0];
  float chi2 = 0;
  chi2 = sub_sparse_map->propa_errors[index];
  Vector3f update; update.setZero();
  for(int iter = 0; iter<n_iter; ++iter)
  {
    int u_r = floor(u);
    int v_r = floor(v);
    if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
      break;

    if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u-u_r;
    float subpix_y = v-v_r;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    float* it_ref = ref_patch;
    float* it_ref_dx = ref_patch_dx;
    float* it_ref_dy = ref_patch_dy;
    float new_chi2 = 0.0;
    Vector3f Jres; Jres.setZero();
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_; 
      for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
      {
        float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
        float res = search_pixel - *it_ref + mean_diff;
        Jres[0] -= res*(*it_ref_dx);
        Jres[1] -= res*(*it_ref_dy);
        Jres[2] -= res;
        new_chi2 += res*res;
      }
    }

    if(iter > 0 && new_chi2 > chi2)
    {
    //   cout << "error increased." << endl;
      u -= update[0];
      v -= update[1];
      break;
    }
    chi2 = new_chi2;

    sub_sparse_map->align_errors[index] = new_chi2;

    update = Hinv * Jres;
    u += update[0];
    v += update[1];
    mean_diff += update[2];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
//         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}

void LidarSelector::FeatureAlignment(cv::Mat img)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    memset(align_flag, 0, length);
    int FeatureAlignmentNum = 0;
       
    for (int i=0; i<total_points; i++) 
    {
        bool res;
        int search_level = sub_sparse_map->search_levels[i];
        Vector2d px_scaled(sub_sparse_map->px_cur[i]/(1<<search_level));
        res = align2D(new_frame_->img_pyr_[search_level], sub_sparse_map->patch_with_border[i], sub_sparse_map->patch[i],
                        20, px_scaled, i);
        sub_sparse_map->px_cur[i] = px_scaled * (1<<search_level);
        if(res)
        {
            align_flag[i] = 1;
            FeatureAlignmentNum++;
        }
    }
}
#endif

float LidarSelector::UpdateState(cv::Mat img, float total_residual, int level) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return 0.;
    StatesGroup old_state = (*state);
    V2D pc; 
    MD(1,2) Jimg;
    MD(2,3) Jdpi;
    MD(1,3) Jdphi, Jdp, JdR, Jdt;
    VectorXd z;
    // VectorXd R;
    bool EKF_end = false;
    /* Compute J */
    float error=0.0, last_error=total_residual, patch_error=0.0, last_patch_error=0.0, propa_error=0.0;
    // MatrixXd H;
    bool z_init = true;
    const int H_DIM = total_points * patch_size_total;//总残差维度=总patch数 * 单个patch像素数
    
    // K.resize(H_DIM, H_DIM);
    z.resize(H_DIM);
    z.setZero();
    // R.resize(H_DIM);
    // R.setZero();

    // H.resize(H_DIM, DIM_STATE);
    // H.setZero();
    H_sub.resize(H_DIM, 6);
    H_sub.setZero();
    
    for (int iteration=0; iteration<NUM_MAX_ITERATIONS; iteration++) 
    {
        // double t1 = omp_get_wtime();
        double count_outlier = 0;
     
        error = 0.0;
        propa_error = 0.0;
        n_meas_ =0;
        M3D Rwi(state->rot_end);
        V3D Pwi(state->pos_end);
        Rcw = Rci * Rwi.transpose();
        Pcw = -Rci*Rwi.transpose()*Pwi + Pci;
        Jdp_dt = Rci * Rwi.transpose();//Pc对位姿t的雅可比,但是缺了*(-1)
        
        M3D p_hat;
        int i;

        for (i=0; i<sub_sparse_map->index.size(); i++) 
        {
            patch_error = 0.0;
            int search_level = sub_sparse_map->search_levels[i];
            int pyramid_level = level + search_level;
            const int scale =  (1<<pyramid_level);
            
            PointPtr pt = sub_sparse_map->voxel_points[i];

            if(pt==nullptr) continue;

            V3D pf = Rcw * pt->pos_ + Pcw;      //匹配patch中心点相机系3d坐标Pc
            pc = cam->world2cam(pf);            //匹配patch在cur帧的像素坐标
            // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
            {
                dpi(pf, Jdpi);                  //uv对相机系坐标Pc的雅可比
                p_hat << SKEW_SYM_MATRX(pf);
            }
            //双线性插值计算 pc 点亚像素精度的 光度
            const float u_ref = pc[0];
            const float v_ref = pc[1];
            const int u_ref_i = floorf(pc[0]/scale)*scale;    //先缩小到塔高层得到对应的 int 像素坐标,再放大还原到0层的得到原始图像坐标
            const int v_ref_i = floorf(pc[1]/scale)*scale;
            const float subpix_u_ref = (u_ref-u_ref_i)/scale; //在0层的插值ratio再缩小到塔高层
            const float subpix_v_ref = (v_ref-v_ref_i)/scale;
            const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);//在塔高层上双线性插值权重
            const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
            const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;
            
            vector<float> P = sub_sparse_map->patch[i];
            for (int x=0; x<patch_size; x++) //这个x实际是竖直的y
            {
                //这里的图像梯度是在cur图上求得的,所以是正向梯度法
                
                //这里求塔高层图像上的patch块梯度. 塔高层又没有计算金字塔图像,所以需要到塔0层取具体的灰度值.                
                //考虑金字塔尺度, 就固定patch块中心在塔0层坐标(u_ref_i,v_ref_i), 然后在塔高层单步移动1个像素,然后投射到塔0层就是移动1*scale个像素
                //      所以塔高层移动(step_x, step_y),对应到塔0层就是(u_ref_i + step_x*scale, v_ref_i + step_y*scale)
                //      确定图像x,y坐标,一维展开坐标= y*width + x
                //      考虑金字塔尺度一维展开就是 = (v_ref_i + step_y*scale)*width + u_ref_i + step_x*scale
                //考虑梯度计算,du要用(x+1,y) (x-1,y)点, dv要用(x,y+1) (x,y-1)点,考虑尺度du要用(x+scale,y) (x-scale,y)点,dv要用(x,y+scale) (x,y-scale)点
                //考虑双线性插值,上面用到的4个点, 以其坐标为左上角点,然后做双线性插值重新得到 4个点坐标再来计算 du,dv

                uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i+x*scale-patch_size_half*scale)*width + u_ref_i-patch_size_half*scale;
                for (int y=0; y<patch_size; ++y, img_ptr+=scale) 
                {
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                    //{
                    float du = 0.5f * ((w_ref_tl*img_ptr[scale] + w_ref_tr*img_ptr[scale*2] + w_ref_bl*img_ptr[scale*width+scale] + w_ref_br*img_ptr[scale*width+scale*2])
                                -(w_ref_tl*img_ptr[-scale] + w_ref_tr*img_ptr[0] + w_ref_bl*img_ptr[scale*width-scale] + w_ref_br*img_ptr[scale*width]));
                    float dv = 0.5f * ((w_ref_tl*img_ptr[scale*width] + w_ref_tr*img_ptr[scale+scale*width] + w_ref_bl*img_ptr[width*scale*2] + w_ref_br*img_ptr[width*scale*2+scale])
                                -(w_ref_tl*img_ptr[-scale*width] + w_ref_tr*img_ptr[-scale*width+scale] + w_ref_bl*img_ptr[0] + w_ref_br*img_ptr[scale]));
                    Jimg << du, dv;             //塔0层得到的梯度
                    Jimg = Jimg * (1.0/scale);  //将梯度缩小到塔高层
                    
                    //这里考虑了camera和imu外参后, 再计算err到imu位姿的雅可比.
                    // [warn:公式可以再推导再验证]
                    Jdphi = Jimg * Jdpi * p_hat;//这个已经是光度err对旋转R李代数的雅可比了!
                    Jdp = -Jimg * Jdpi;
                    JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR;
                    Jdt = Jdp * Jdp_dt;         //光度err对位姿t的雅可比[ok]
                    //}
                    //计算光度残差, 还是要双线性插值得到cur帧(x,y)处的灰度 - ref的patch对应尺度对应位置的灰度
                    double res = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[scale] + w_ref_bl*img_ptr[scale*width] + w_ref_br*img_ptr[scale*width+scale]  - P[patch_size_total*level + x*patch_size+y];
                    //第i个patch了,前面已经有i*patch_size_total残差计算好了,这里是第i个patch的(x,y)位置的残差
                    z(i*patch_size_total+x*patch_size+y) = res;
                    // float weight = 1.0;
                    // if(iteration > 0)
                    //     weight = weight_function_->value(res/weight_scale_); 
                    // R(i*patch_size_total+x*patch_size+y) = weight;       
                    patch_error +=  res*res;
                    n_meas_++;
                    // H.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR*weight, Jdt*weight;
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                    
                    //H_sub是具体patch的具体(x,y)位置的残差对pose6d的雅可比
                    H_sub.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR, Jdt;
                }
            }  

            sub_sparse_map->errors[i] = patch_error;
            error += patch_error;
        }

        // computeH += omp_get_wtime() - t1;

        error = error/n_meas_;

        // double t3 = omp_get_wtime();

        // 下面是Esikf的后验刷新公式,可以再推导再验证
        if (error <= last_error) 
        {
            old_state = (*state);
            last_error = error;

            // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov;
            // auto vec = (*state_propagat) - (*state);
            // G = K*H;
            // (*state) += (-K*z + vec - G*vec);

            auto &&H_sub_T = H_sub.transpose();
            H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;
            MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
            auto &&HTz = H_sub_T * z;
            // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
            auto vec = (*state_propagat) - (*state);
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
            auto solution = - K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);
            (*state) += solution;
            auto &&rot_add = solution.block<3,1>(0,0);
            auto &&t_add   = solution.block<3,1>(3,0);

            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
            {
                EKF_end = true;
            }
        }
        else
        {
            (*state) = old_state;
            EKF_end = true;
        }

        // ekf_time += omp_get_wtime() - t3;

        if (iteration==NUM_MAX_ITERATIONS || EKF_end) 
        {
            break;
        }
    }
    return last_error;
} 
//单纯就是把预测imu的pose用外参转换到相机pose,设入new_frame_
void LidarSelector::updateFrameState(StatesGroup state)
{
    M3D Rwi(state.rot_end);
    V3D Pwi(state.pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci*Rwi.transpose()*Pwi + Pci;
    new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}
//将当前帧新观测的patch添加进对应Point中
// 1)用后验刷新后的pose,重新计算point到当前帧的投影像素pc
// 2)add check: 距离上个patch的帧pose超过50cm或者10度, 距离上个patch的投影像素超过40
// 3)delete冗余 patch: 单个point最大观测数20个, 优先删除距离当前帧平移最远的patch观测
void LidarSelector::addObservation(cv::Mat img)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;

    for (int i=0; i<total_points; i++) 
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        if(pt==nullptr) continue;
        V2D pc(new_frame_->w2c(pt->pos_));
        SE3 pose_cur = new_frame_->T_f_w_;
        bool add_flag = false;
        // if (sub_sparse_map->errors[i]<= 100*patch_size_total && sub_sparse_map->errors[i]>0)
        {
            //TODO: condition: distance and view_angle 
            // Step 1: time
            FeaturePtr last_feature =  pt->obs_.back();
            // if(new_frame_->id_ >= last_feature->id_ + 20) add_flag = true;

            // Step 2: delta_pose
            SE3 pose_ref = last_feature->T_f_w_;
            SE3 delta_pose = pose_ref * pose_cur.inverse();
            double delta_p = delta_pose.translation().norm();
            double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));            
            if(delta_p > 0.5 || delta_theta > 10) add_flag = true;

            // Step 3: pixel distance
            Vector2d last_px = last_feature->px;
            double pixel_dist = (pc-last_px).norm();
            if(pixel_dist > 40) add_flag = true;
            
            // Maintain the size of 3D Point observation features.
            if(pt->obs_.size()>=20)
            {
                FeaturePtr ref_ftr;
                pt->getFurthestViewObs(new_frame_->pos(), ref_ftr);
                pt->deleteFeatureRef(ref_ftr);
                // ROS_WARN("ref_ftr->id_ is %d", ref_ftr->id_);
            } 
            if(add_flag)
            {
                pt->value = vk::shiTomasiScore(img, pc[0], pc[1]);
                Vector3d f = cam->cam2world(pc);
                FeaturePtr ftr_new(new Feature(pc, f, new_frame_->T_f_w_, pt->value, sub_sparse_map->search_levels[i])); 
                ftr_new->img = new_frame_->img_pyr_[0];
                ftr_new->id_ = new_frame_->id_;
                // ftr_new->ImgPyr.resize(5);
                // for(int i=0;i<5;i++) ftr_new->ImgPyr[i] = new_frame_->img_pyr_[i];
                pt->addFrameRef(ftr_new);      
            }
        }
    }
}

void LidarSelector::ComputeJ(cv::Mat img) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    float error = 1e10;
    float now_error = error;

    for (int level=2; level>=0; level--) 
    {
        now_error = UpdateState(img, error, level);
    }
    if (now_error < error)
    {
        state->cov -= G*state->cov;
    }
    updateFrameState(*state);
}

void LidarSelector::display_keypatch(double time)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    for(int i=0; i<total_points; i++)
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        V2D pc(new_frame_->w2c(pt->pos_));
        cv::Point2f pf;
        pf = cv::Point2f(pc[0], pc[1]); 
        if (sub_sparse_map->errors[i]<8000) // 5.5
            cv::circle(img_cp, pf, 6, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
        else
            cv::circle(img_cp, pf, 6, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }   
    std::string text = std::to_string(int(1/time))+" HZ";
    cv::Point2f origin;
    origin.x = 20;
    origin.y = 20;
    cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
}

V3F LidarSelector::getpixel(cv::Mat img, V2D pc) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]); 
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref-u_ref_i);
    const float subpix_v_ref = (v_ref-v_ref_i);
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    uint8_t* img_ptr = (uint8_t*) img.data + ((v_ref_i)*width + (u_ref_i))*3;
    float B = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[0+3] + w_ref_bl*img_ptr[width*3] + w_ref_br*img_ptr[width*3+0+3];
    float G = w_ref_tl*img_ptr[1] + w_ref_tr*img_ptr[1+3] + w_ref_bl*img_ptr[1+width*3] + w_ref_br*img_ptr[width*3+1+3];
    float R = w_ref_tl*img_ptr[2] + w_ref_tr*img_ptr[2+3] + w_ref_bl*img_ptr[2+width*3] + w_ref_br*img_ptr[width*3+2+3];
    V3F pixel(B,G,R);
    return pixel;
}

//进入img的处理流程
void LidarSelector::detect(cv::Mat img, PointCloudXYZI::Ptr pg) 
{
    if(width!=img.cols || height!=img.rows)
    {
        // std::cout<<"Resize the img scale !!!"<<std::endl;
        double scale = 0.5;
        cv::resize(img,img,cv::Size(img.cols*scale,img.rows*scale),0,0,CV_INTER_LINEAR);
    }
    img_rgb = img.clone();
    img_cp = img.clone();
    cv::cvtColor(img,img,CV_BGR2GRAY);

    new_frame_.reset(new Frame(cam, img.clone()));
    //单纯就是把预测imu的pose用外参转换到相机pose,设入new_frame_
    updateFrameState(*state);

    if(stage_ == STAGE_FIRST_FRAME && pg->size()>10)
    {
        new_frame_->setKeyframe();//填充设置5个特征
        stage_ = STAGE_DEFAULT_FRAME;
    }

    double t1 = omp_get_wtime();
    // 当前帧图像划分grid网格后, 检索体素地图并做patch匹配,结果缓存进 sub_sparse_map. [本步骤前sub_sparse_map被实时清空]
    addFromSparseMap(img, pg);

    double t3 = omp_get_wtime();
    // lidar点投影到当前网格,新建 Point点, Feature观测, 并将新建Point点添加进全局体素地图 feat_map
    // [单个grid内已经匹配到的体素点,如果响应值被超过,也会直接新建point]
    // [本步骤新新建点到 全局 feat_map, 但是没有添加到局部 sub_sparse_map]
    addSparseMap(img, pg);

    double t4 = omp_get_wtime();
    
    // computeH = ekf_time = 0.0;
    //利用光度误差作后验刷新.[再细看,再验证]
    ComputeJ(img);

    double t5 = omp_get_wtime();

    //后验刷新了当前帧pose, 基于此,将当前帧新观测的patch添加进对应Point中
    addObservation(img);
    
    double t2 = omp_get_wtime();
    
    frame_count ++;
    ave_total = ave_total * (frame_count - 1) / frame_count + (t2 - t1) / frame_count;

    printf("[ VIO ]: time: addFromSparseMap: %.6f addSparseMap: %.6f ComputeJ: %.6f addObservation: %.6f total time: %.6f ave_total: %.6f.\n"
    , t3-t1, t4-t3, t5-t4, t2-t5, t2-t1, ave_total);

    display_keypatch(t2-t1);
} 

} // namespace lidar_selection