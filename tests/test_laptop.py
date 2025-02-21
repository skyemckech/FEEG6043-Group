from laptop import LaptopPilot
import openpyxl
from src.Libraries.math_feeg6043 import Vector, Transpose, m2l
import numpy as np

def test_laptop():
    Laptop = LaptopPilot(True)
    Laptop.infinite_loop
    assert Laptop.robot_ip == '127.0.0.1'

class createExcelFile:
        def __init__(self):
            self.workbook = openpyxl.Workbook()
            self.worksheet = self.workbook.active
            self.dataLine = []
        def extend_data(self, data):
            self.dataLine.extend(data)
        def export_to_excel(self,filename = "data.xslx"):
            self.worksheet.append(self.dataLine)       
            self.workbook.save(filename)

def test_createExcelFile():
    testSheet = createExcelFile()
    p_example = Vector(3)
    p_example[0,0] = 1
    p_example[1,0] = 2
    p_example[2,0] = 3
    list2=[4,5,6]
    list3=[7,8,9]
    testSheet.extend_data([2,3])
    testSheet.extend_data([p_example[0,0]])
    testSheet.extend_data(list3)
    testSheet.export_to_excel("MEGAGOONEAAAH.xlsx")

def test_type_checker():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    list = [1,2,3]
    matrix = Vector(4)
    assert type(array) == type(np.array([]))
    assert type(list) == type([])
    assert type(matrix) == type(np.array([]))
    