#!/usr/bin/python
# coding=utf-8

import os
import numpy as np
import time
from PyQt5 import QtWidgets, QtCore
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.util import numpy_support
import utils
from reshaper import Reshaper
from mesh_loader import DepthMeshLoader

# Clase para señales de slider personalizado
class IndexedQSlider(QtWidgets.QSlider):
  valueChangeForwarded = QtCore.pyqtSignal(int, int, int, int)
  def __init__(self, sliderID, orientation, parent=None):
    QtWidgets.QSlider.__init__(self, orientation, parent)
    self.sliderID = sliderID
    self.valueChanged.connect(self.valueChangeForwarder)

  # Emitir señal de cambio de valor personalizada
  def valueChangeForwarder(self, val):
    self.valueChangeForwarded.emit(
      self.sliderID, val, self.minimum(), self.maximum())

# Acción personalizada para menú
class myAction(QtWidgets.QAction):
  myact = QtCore.pyqtSignal(int)
  def __init__(self, _id, *args):
    QtWidgets.QAction.__init__(self, *args)
    self._id = _id
    self.triggered.connect(self.emitSelect)

  def emitSelect(self):
    self.myact.emit(self._id)

# Widget para renderizar el modelo 3D con VTK
class MayaviQWidget(QtWidgets.QWidget):
  def __init__(self, parent):
    QtWidgets.QWidget.__init__(self, parent)
    layout = QtWidgets.QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    
    # Crear el widget VTK
    self.vtk_widget = QVTKRenderWindowInteractor(self)
    layout.addWidget(self.vtk_widget)
    
    # Configurar el renderizador VTK
    self.renderer = vtk.vtkRenderer()
    self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
    self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
    
    # Crear la fuente de datos para la malla
    self.mesh_source = vtk.vtkPolyData()
    self.points = vtk.vtkPoints()
    self.cells = vtk.vtkCellArray()
    
    # Crear el mapper y actor para la malla
    self.mesh_mapper = vtk.vtkPolyDataMapper()
    self.mesh_actor = vtk.vtkActor()
    self.mesh_actor.SetMapper(self.mesh_mapper)
    
    # Configurar el aspecto de la malla
    self.mesh_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    self.mesh_actor.GetProperty().SetSpecular(0.3)
    self.mesh_actor.GetProperty().SetSpecularPower(20)
    
    # Añadir el actor al renderizador
    self.renderer.AddActor(self.mesh_actor)
    self.renderer.SetBackground(0.1, 0.1, 0.1)
    
    # Inicializar los modelos del cuerpo
    self.bodies = {"female": Reshaper(label="female"), "male": Reshaper(label="male")}
    self.body = self.bodies["female"]
    self.flag_ = 0

    self.vertices = self.body.mean_vertex
    self.normals = self.body.normals
    self.facets = self.body.facets
    self.input_data = np.zeros((utils.M_NUM, 1))
    
    # Loader for external meshes
    self.external_loader = DepthMeshLoader()
    self.is_external_mesh = False
    self.external_polydata = None
    
    # --- CONFIGURACIÓN DE NUBE DE PUNTOS (DEPTH MAP) ---
    self.pc_actor = vtk.vtkActor()
    self.pc_mapper = vtk.vtkPolyDataMapper()
    self.pc_actor.SetMapper(self.pc_mapper)
    self.pc_actor.GetProperty().SetPointSize(3)  # Tamaño del punto
    self.pc_actor.GetProperty().SetColor(0.0, 1.0, 1.0)  # Cyan
    self.pc_actor.VisibilityOff()  # Oculto por defecto
    self.renderer.AddActor(self.pc_actor)
    # ---------------------------------------------------
    
    # Iniciar el renderizado
    self.interactor.Initialize()
    
    # Actualizar visualización
    self.update()

  def update(self):
    if self.is_external_mesh:
        if self.external_polydata:
            # Calcular normales
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(self.external_polydata)
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOn()
            normals.Update()
            
            # Actualizar el mapper
            self.mesh_mapper.SetInputData(normals.GetOutput())
    else:
        [self.vertices, self.normals, self.facets] = \
            self.body.mapping(self.input_data, self.flag_)
        self.vertices = self.vertices.astype('float32')
        
        # Actualizar los puntos de la malla
        points = vtk.vtkPoints()
        points.SetData(vtk.util.numpy_support.numpy_to_vtk(self.vertices, deep=True))
        
        cells = vtk.vtkCellArray()
        # En VTK 9, podemos usar una forma más eficiente para las celdas
        connectivity = vtk.util.numpy_support.numpy_to_vtk(self.facets.flatten(), deep=True, array_type=vtk.VTK_ID_TYPE)
        offsets = vtk.util.numpy_support.numpy_to_vtk(np.arange(0, len(self.facets) * 3 + 1, 3), deep=True, array_type=vtk.VTK_ID_TYPE)
        cells.SetData(offsets, connectivity)
        
        # Actualizar la fuente de datos
        self.mesh_source = vtk.vtkPolyData()
        self.mesh_source.SetPoints(points)
        self.mesh_source.SetPolys(cells)
        
        # Calcular normales
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(self.mesh_source)
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOn()
        normals.Update()
        
        # Actualizar el mapper
        self.mesh_mapper.SetInputData(normals.GetOutput())
    
    # Renderizar
    self.vtk_widget.GetRenderWindow().Render()
    
    # Ajustar la cámara
    self.renderer.ResetCamera()

  def select_mode(self, label="female", flag=0):
    self.is_external_mesh = False
    self.body = self.bodies[label]
    self.flag_ = flag
    self.update()

  def load_external_mesh(self, filepath):
    """Carga y muestra una malla desde un archivo .npy con filtrado y rotación"""
    depth_image = self.external_loader.load_npy(filepath)
    if depth_image is not None:
        # Definir una rotación base si es necesario (ej: 90 grados en X para que esté parado)
        # alpha = np.radians(90)
        # R = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
        
        # O rotación de 180 en Y si está de espaldas
        # beta = np.radians(180)
        # R = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        
        # Por ahora probamos sin rotación pero con filtrado de ruido (max_edge_dist)
        # El usuario menciona que está rotada, así que aplicamos una rotación de ejemplo
        # que suele funcionar para corregir mallas de profundidad (voltear Y y Z)
        R = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])

        # Filtrado de profundidad: asumir que el humano está en el rango central
        # Si el fondo es 0, eso ya lo maneja el loader. 
        # Podemos añadir un umbral de borde para eliminar estiramientos
        vertices, facets = self.external_loader.convert_to_mesh(
            depth_image, 
            max_edge_dist=15.0,  # Este valor rompe los "curtains"
            rotation=R
        )
        
        self.external_polydata = self.external_loader.create_vtk_polydata(vertices, facets)
        self.is_external_mesh = True
        self.update()
        return True
    return False

  def update_depth_cloud(self, numpy_points):
    """
    Recibe un array (N, 3) de puntos y actualiza el actor VTK.
    Usa numpy_support para transferencia eficiente de datos.
    """
    if numpy_points is None or len(numpy_points) == 0:
        self.pc_actor.VisibilityOff()
        self.vtk_widget.GetRenderWindow().Render()
        return

    # 1. Convertir numpy array a vtkPoints eficientemente
    vtk_points = vtk.vtkPoints()
    vtk_data_array = numpy_support.numpy_to_vtk(
        num_array=numpy_points.astype(np.float32), 
        deep=True, 
        array_type=vtk.VTK_FLOAT
    )
    vtk_points.SetData(vtk_data_array)

    # 2. Crear PolyData
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    # 3. Usar vtkVertexGlyphFilter para renderizado masivo de puntos
    vertex_filter = vtk.vtkVertexGlyphFilter()
    vertex_filter.SetInputData(polydata)
    vertex_filter.Update()

    # 4. Asignar al mapper
    self.pc_mapper.SetInputConnection(vertex_filter.GetOutputPort())

    # 5. Hacer visible y refrescar
    self.pc_actor.VisibilityOn()
    self.renderer.ResetCamera()
    self.vtk_widget.GetRenderWindow().Render()
    print(f"[**] Depth cloud renderizado: {len(numpy_points)} puntos")

  def toggle_depth_cloud(self, visible=True):
    """Alternar visibilidad de la nube de puntos."""
    if visible:
        self.pc_actor.VisibilityOn()
    else:
        self.pc_actor.VisibilityOff()
    self.vtk_widget.GetRenderWindow().Render()

  def sliderForwardedValueChangeHandler(self, sliderID, val, minVal, maxVal):
    x = val / 10.0
    self.input_data[sliderID] = x
    start = time.time()
    self.update()
    print(' [**] update body in %f s' % (time.time() - start))

  def save(self):
    utils.save_obj("result.obj", self.vertices, self.facets+1)
    output = np.array(utils.calc_measure(self.body.cp, self.vertices, self.facets))
    for i in range(0, utils.M_NUM):
      print("%s: %f" % (utils.M_STR[i], output[i, 0]))

  def predict(self, data):
    mask = np.zeros((utils.M_NUM, 1), dtype=bool)
    for i in range(0, data.shape[0]):
      if data[i, 0] != 0:
        data[i, 0] -= self.body.mean_measure[i, 0]
        data[i, 0] /= self.body.std_measure[i, 0]
        mask[i, 0] = 1
    self.input_data = self.body.get_predict(mask, data)
    self.update()
    measure = self.body.mean_measure + self.input_data*self.body.std_measure
    return [self.input_data, measure]
