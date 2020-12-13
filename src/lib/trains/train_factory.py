from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .ctdet import CtdetTrainer
from .ddd import DddTrainer
from .exdet import ExdetTrainer
from .multi_pose import MultiPoseTrainer
from .CenterLandmark_trainer import CenterLandmarkTrainer
from .clothlandmark_trainer import clothLandmarkTrainer
from .gridneighbordet import GridneighbordetTrainer
from .ctdet_doublehm import CtdetTrainer_Doublehm
from .ctdet_esphm_trainer import CtdetTrainer_esphm

train_factory = {
  'exdet': ExdetTrainer, 
  'ddd': DddTrainer,
  'ctdet': CtdetTrainer,
  'multi_pose': MultiPoseTrainer,
  'landmark': CenterLandmarkTrainer,
  'cloth': clothLandmarkTrainer,
  'gridneighbordet': GridneighbordetTrainer,
  'ctdet_Dhm':CtdetTrainer_Doublehm,
  'ctdet_esphm': CtdetTrainer_esphm
}
