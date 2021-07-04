import os
import sys
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
import srv1_A_create_raw_water_csv.data_model as dm     # pylint: disable=import-error


dataModel = dm.DataModel()

dataModel.convert_nc_water_to_raw_csv()
