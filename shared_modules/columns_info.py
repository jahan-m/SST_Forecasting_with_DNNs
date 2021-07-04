

class ColumnsInfo:
    
    def __init__(self):
        self.DATE = "date"
        self.DATE_TYPE = str
        self.AIR = "air_temp"
        self.AIR_TYPE = float
        self.WATER = "water_temp"
        self.WATER_TYPE = float
    
    
    
    def date_column(self, sz=False):
        if sz:
            return self.DATE
        else:
            return [self.DATE]
    
    def air_column(self, sz=False):
        if sz:
            return self.AIR
        else:
            return [self.AIR]
    
    def water_column(self, sz=False):
        if sz:
            return self.WATER
        else:
            return [self.WATER]
    
    
    
    def date_air_columns(self):
        return [self.DATE, self.AIR]
    
    def date_water_columns(self):
        return [self.DATE, self.WATER]
    
    def air_water_columns(self):
        return [self.AIR, self.WATER]
    
    def date_air_water_columns(self):
        return [self.DATE, self.AIR, self.WATER]
    
    
    def date_dtypes(self):
        return {self.DATE:self.DATE_TYPE}
    
    def date_air_dtypes(self):
        return {self.DATE:self.DATE_TYPE, self.AIR:self.AIR_TYPE}
    
    def date_water_dtypes(self):
        return {self.DATE:self.DATE_TYPE, self.WATER:self.WATER_TYPE}

    def date_air_water_dtypes(self):
        return {self.DATE:self.DATE_TYPE, self.AIR:self.AIR_TYPE, self.WATER:self.WATER_TYPE}
