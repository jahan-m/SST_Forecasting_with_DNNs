import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import srv1_B_create_raw_air_csv.data_model as dm       # pylint: disable=import-error


dataModel = dm.DataModel()

dataModel.convert_dataset_air_to_raw_csv()
