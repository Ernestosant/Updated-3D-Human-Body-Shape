import numpy as np
import os
import open3d as o3d


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "release_model")
V_NUM = 12500
F_NUM = 25000
M_NUM = 19
D_BASIS_NUM = 10
V_BASIS_NUM = 10

M_STR = ["weight", "height", "neck", "chest",
  "belly button waist", "gluteal hip",
  "neck shoulder elbow wrist", "crotch knee floor",
  "across back shoulder neck", "neck to gluteal hip",
  "natural waist", "max. hip", "natural waist rise",
  "shoulder to midhand", "upper arm", "wrist",
  "outer natural waist to floor", "knee", "max. thigh"]

P2M = [[0, 1, 7, 16, 17], [0, 1, 6, 13, 14], [6, 13, 15],
  [0, 1, 7, 16, 17], [6, 13, 15],
  [0, 1, 4, 5, 7, 9, 11, 12, 16, 18],
  [0, 1, 3, 4, 5, 9, 10, 11, 12, 16],
  [0, 1, 5, 7, 16, 17, 18], [0, 1, 6, 13, 15],
  [0, 1, 2, 3, 6, 8, 9], [16],
  [0, 1, 2], [0, 1, 6, 13, 14], [16],
  [0, 1, 5, 7, 16, 17, 18], [0, 1, 6, 13, 15]]

PART = [(0.0, np.float64(0.66666700000000001), 1.0),
  (np.float64(0.66666700000000001), np.float64(0.66666700000000001), 0.0),
  (np.float64(0.66666700000000001), 1.0, 0.0),
  (1.0, np.float64(0.66666700000000001), 0.0),
  (np.float64(0.66666700000000001),0.0, 0.0),
  (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0),
  (0.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0),
  (np.float64(0.32941199999999998), 0.0, 0.494118),
  (1.0, 1.0, 1.0),
  (np.float64(0.66666700000000001), 0.0, 1.0),
  (0.0, np.float64(0.32941199999999998), 0.0)]

# save obj file
def save_obj(filename, v, f):
  file = open(filename, 'w')
  for i in range(0, v.shape[0]):
    file.write('v %f %f %f\n'%(v[i][0], v[i][1], v[i][2]))
  for i in range(0, f.shape[0]):
    file.write('f %d %d %d\n'%(f[i][0], f[i][1], f[i][2]))
  file.close()
  tmp = v[:, 2]
  print('[**] save obj file in {}, height: {}'.format(filename, tmp.max() - tmp.min()))

# calculate the corresponding deformation from the input vertex
def get_deform(vertex, facet, d_inv_mean):
  deform = np.zeros((F_NUM, 9))
  for i in range(0, F_NUM):
    v = [k - 1 for k in facet[i, :]]
    v1 = vertex[v[0], :]
    v2 = vertex[v[1], :]
    v3 = vertex[v[2], :]
    Q =  assemble_face(v1, v2, v3).dot(d_inv_mean[i])
    deform[i, :] = Q.flat
  return deform

# import the 4th point of the triangle, and calculate the deformation
def assemble_face(v1, v2, v3):
  v21 = np.array((v2 - v1))
  v31 = np.array((v3 - v1))
  v41 = np.cross(list(v21.flat), list(v31.flat))
  v41 /= np.sqrt(np.linalg.norm(v41))
  return np.column_stack((v21, np.column_stack((v31, v41))))

# calculate measure data from given vertex by control points
def calc_measure(cp, vertex, facet):
  measure_list = []
  # clac weight
  vol = 0.0
  kHumanbodyIntensity = 1026.0
  for i in range(0, F_NUM):
    f = [c - 1 for c in facet[i, :]]
    v0 = vertex[f[0], :]
    v1 = vertex[f[1], :]
    v2 = vertex[f[2], :]
    vol += np.cross(v0, v1).dot(v2)
  vol = abs(vol) / 6.0
  weight = kHumanbodyIntensity * vol
  # weight = weight**(1.0 / 3.0) * 1000
  measure_list.append(weight)
  # calc other measures
  for measure in cp:
    length = 0.0
    p2 = vertex[int(measure[0][1]), :]
    for i in range(1, len(measure)):
      p1 = p2
      if measure[i][0] == 1:
        p2 = vertex[int(measure[i][1]), :]
      elif measure[i][0] == 2:
        p2 = vertex[int(measure[i][1]), :] * measure[i][3] + \
          vertex[int(measure[i][2]), :] * measure[i][4]
      else:
        p2 = vertex[int(measure[i][1]), :] * measure[i][4] + \
          vertex[int(measure[i][2]), :] * measure[i][5] + \
          vertex[int(measure[i][3]), :] * measure[i][6]
      length += np.sqrt(np.sum((p1 - p2)**2.0))
    measure_list.append(length * 1000)
  return np.array(measure_list).reshape(M_NUM, 1)


def load_depth_as_pointcloud(filename, fx=1000.0, fy=1000.0, depth_scale=1000.0, stride=2):
    """
    Carga un archivo .npy de profundidad y devuelve los vértices (N, 3)
    listos para ser renderizados por VTK.
    
    Args:
        filename: Ruta al archivo .npy
        fx, fy: Parámetros intrínsecos focales (aproximados si no se conocen)
        depth_scale: Factor de escala para normalizar profundidad
        stride: Submuestreo para reducir puntos (2 = la mitad de resolución)
    
    Returns:
        numpy.ndarray: Array de puntos (N, 3) o None si hay error
    """
    if not os.path.exists(filename):
        print(f"Archivo no encontrado: {filename}")
        return None

    try:
        # 1. Cargar datos
        depth_data = np.load(filename).astype(np.float32)
        
        # 2. Validar que sea 2D
        if depth_data.ndim != 2:
            print(f"Error: Se esperaba imagen 2D, recibido shape {depth_data.shape}")
            return None
        
        # 3. Convertir a Geometría Open3D
        height, width = depth_data.shape
        img_depth = o3d.geometry.Image(depth_data)
        
        # 4. Intrínsecos aproximados (centro de imagen)
        cx, cy = width / 2, height / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        
        # 5. Generar Nube de Puntos
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            img_depth, 
            intrinsic,
            depth_scale=depth_scale, 
            depth_trunc=10.0,  # Truncar puntos muy lejanos
            stride=stride  # Reducir resolución para rendimiento
        )
        
        # 6. Filtrar outliers estadísticos (mejora visualización)
        if len(pcd.points) > 100:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 7. Centrar la nube de puntos
        pcd.translate(-pcd.get_center())
        
        # 8. Devolver como numpy array
        points = np.asarray(pcd.points)
        print(f"[**] Depth map cargado: {len(points)} puntos desde {filename}")
        return points
        
    except Exception as e:
        print(f"Error procesando depth map: {e}")
        import traceback
        traceback.print_exc()
        return None
