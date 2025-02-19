from laptop import LaptopPilot

def test_laptop():
    #Arrange
    simulated = True
    #Act
    laptop = LaptopPilot(simulated)
    #Assert 
    assert laptop.robot_ip == "127.0.0.1"
