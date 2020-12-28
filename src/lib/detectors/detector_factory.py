from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .exdet import ExdetDetector
from .ddd import DddDetector
from .ctdet import CtdetDetector
from .multi_pose import MultiPoseDetector
from .landmark_detector import LandmarkDetector
from .deepfashion2_detector import ClothDetector
from .gridneighbordet_detector import GridneighbordetDetector
from .ctdet_Dhm import CtdetDetector_Doublehm
from .ctdet_esphm import CtdetDetector_esphm
from .vehicledet_detector import VehicledetDetector
from .vehicledetSinAngle_detector import VehicledetSinAngleDetector

detector_factory = {
  'exdet': ExdetDetector, 
  'ddd': DddDetector,
  'ctdet': CtdetDetector,
  'multi_pose': MultiPoseDetector,
  'landmark' : LandmarkDetector,
  'cloth' : ClothDetector,
  'gridneighbordet': GridneighbordetDetector,
  'ctdet_Dhm':CtdetDetector_Doublehm,
  'ctdet_esphm':CtdetDetector_esphm,
  'vehicle_det' : VehicledetDetector,
  'vehicle_det_SinAngle' : VehicledetSinAngleDetector
}
