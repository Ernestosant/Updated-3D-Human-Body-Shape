#!/usr/bin/python
# coding=utf-8

import os
import numpy as np
import time
from PyQt5 import QtWidgets, QtCore
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import utils
from reshaper import Reshaper

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
    
    # Iniciar el renderizado
    self.interactor.Initialize()
    
    # Actualizar visualización
    self.update()

  def update(self):
    [self.vertices, self.normals, self.facets] = \
        self.body.mapping(self.input_data, self.flag_)
    self.vertices = self.vertices.astype('float32')
    
    # Actualizar los puntos de la malla
    self.points = vtk.vtkPoints()
    for i in range(len(self.vertices)):
        self.points.InsertNextPoint(self.vertices[i][0], self.vertices[i][1], self.vertices[i][2])
    
    # Actualizar las celdas (triángulos) de la malla
    self.cells = vtk.vtkCellArray()
    for i in range(len(self.facets)):
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, int(self.facets[i][0]))
        triangle.GetPointIds().SetId(1, int(self.facets[i][1]))
        triangle.GetPointIds().SetId(2, int(self.facets[i][2]))
        self.cells.InsertNextCell(triangle)
    
    # Actualizar la fuente de datos
    self.mesh_source = vtk.vtkPolyData()
    self.mesh_source.SetPoints(self.points)
    self.mesh_source.SetPolys(self.cells)
    
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
    self.body = self.bodies[label]
    self.flag_ = flag
    self.update()

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
