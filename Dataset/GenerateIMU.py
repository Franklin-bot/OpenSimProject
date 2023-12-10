import xml.etree.ElementTree as ET
import numpy as np

filepath = "/Users/FranklinZhao/OpenSimProject/Simulation/Models/Rajapogal_2015/imu_data/model_default_arm_IMU.osim"


def add_imu(component_set, imu_name, socket_frame):
    imu = ET.SubElement(component_set, 'IMU', name=imu_name)
    socket_frame_element = ET.SubElement(imu, 'socket_frame')
    socket_frame_element.text = socket_frame

def add_imu_frame(root, body, name, translation):
    body = root.find(f".//Body[@name=\'{body}\']")
    components = body.find('components')

    physical_offset_frame = ET.SubElement(components, 'PhysicalOffsetFrame', name=name)
    
    frame_geometry = ET.SubElement(physical_offset_frame, 'FrameGeometry', name='frame_geometry')
    
    socket_frame = ET.SubElement(frame_geometry, 'socket_frame')
    socket_frame.text = '..'
    
    scale_factors = ET.SubElement(frame_geometry, 'scale_factors')
    scale_factors.text = '0.20000000000000001 0.20000000000000001 0.20000000000000001'
    
    socket_parent = ET.SubElement(physical_offset_frame, 'socket_parent')
    socket_parent.text = '..'
    
    translation_element = ET.SubElement(physical_offset_frame, 'translation')
    translation_element.text = arrayToString(translation)
    
    orientation = ET.SubElement(physical_offset_frame, 'orientation')
    orientation.text = '0 1.57 1.57'

def modifyIMUTranslation(tree, body, imuName, modifier):
    root = tree.getroot()
    xml_header = '<?xml version="1.0" encoding="UTF-8" ?>'
    modifiers = [[0, 0, modifier], [0, modifier, modifier], [0, modifier, 0], [0, modifier, -modifier], [0, 0, -modifier], [0, -modifier, -modifier], [0, -modifier, 0], [0, -modifier, modifier]]
    component_set = root.find(".//ComponentSet[@name='componentset']/objects")

    curr_position = root.find(f".//Body[@name='{body}']/components/PhysicalOffsetFrame[@name='{imuName}']/translation")
    curr_position = curr_position.text
# Add a new IMU to ulna_l socket frame

    for i, m, in enumerate(modifiers):
        new_imu_name = f"{imuName}{i+1}"
        new_imu_position = np.array(parseStringArray(curr_position)) + np.array(m)
        print(arrayToString(new_imu_position))
        new_imu_frame = f"{new_imu_name}_frame"
        add_imu_frame(root, body, new_imu_frame, new_imu_position)
        add_imu(component_set, new_imu_name, f'/bodyset/{body}/'+new_imu_name+'_frame')


# Save the modified XML to a new file
    return tree
    

def parseStringArray(str):
    curr = ""
    res = []
    for c in str:
        if c == ' ':
            res.append(float(curr))
            curr = ""
        else:
            curr+=c
    res.append(float(curr))
    return res

def arrayToString(arr):
    return " ".join(str(f) for f in arr)


translation = 0.05
tree = ET.parse(filepath)
tree = modifyIMUTranslation(tree, "ulna_l", "ulna_l_imu", translation)
tree = modifyIMUTranslation(tree, "ulna_r", "ulna_r_imu", translation)
tree = modifyIMUTranslation(tree, "humerus_l", "humerus_l_imu", translation)
tree = modifyIMUTranslation(tree, "humerus_r", "humerus_r_imu", translation)
tree.write('modified_xml_file3.osim')







