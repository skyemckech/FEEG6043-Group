from laptop import LaptopPilot
import openpyxl
from src.Libraries.math_feeg6043 import Vector

def test_laptop():
    Laptop = LaptopPilot(True)
    Laptop.infinite_loop
    assert Laptop.robot_ip == '127.0.0.1'

class createExcelFile:
    def __init__(self):
        self.workbook = openpyxl.Workbook()
        self.worksheet = self.workbook.active
    def export_to_excel(self,data,filename = "MEGAGOONBEEAAAHHH.xlsx"):       
        self.worksheet.append(data)
        self.workbook.save(filename)

def test_createExcelFile():
    testSheet = createExcelFile()
    p_robot = Vector[3]
    p_robot[0,0] = 1
    p_robot[1,0] = 2
    p_robot[2,0] = 3
    testSheet.export_to_excel(str(p_robot)) 