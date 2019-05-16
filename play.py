import xml.etree.ElementTree as ET 
import os

def function(tree,filename):
    root = tree.getroot()
    for child in root:
        if child.tag == 'holothurian':
            child.tag = 'object'
        elif child.tag == 'folder':
            child.text = 'VOC2007'
        elif child.tag == "path":
            tmp = child.text
            tmp = (tmp.split('/')[-1]).split('.')[-2]
            tmp = 'JPEGImages/' + tmp
            print(tmp)
            child.text = tmp
for filename in os.listdir('./Annotations'):
    tree = ET.parse('./Annotations/'+str(filename))
    function(tree,filename)
    tree.write('./After/'+filename)
    