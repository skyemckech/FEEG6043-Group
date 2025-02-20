from laptop import LaptopPilot

def test_laptop():
    Laptop = LaptopPilot(True)
    Laptop.infinite_loop
    assert Laptop.robot_ip == '127.0.0.1'

# def test_createExcelFile():
#     Laptop = LaptopPilot(True)
#     createExcelFile = LaptopPilot.createExcelFile()

# def test_innerclasses():
#     inner_instance = OuterClass.InnerClass()
#     print(inner_instance.greet())

import 
if __name__ == "__main__":
    print("OuterClass available:", hasattr(class_definer, "OuterClass"))